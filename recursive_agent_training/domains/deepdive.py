"""DeepDive web tools with caching, bounded retries, and service metrics."""

from __future__ import annotations

import asyncio
import hashlib
import html
from html.parser import HTMLParser
import ipaddress
import json
import os
from pathlib import Path
import re
import socket
import threading
import time
import urllib.parse
import urllib.request
from typing import Any

from recursive_agent_harness.tools import ToolRegistry
from recursive_agent_training.config import SearchServiceConfig


SEARCH_CACHE_VERSION = "deepdive-search-v5-crossref-doi-title"
PAGE_CACHE_NAMESPACE = "pages-v2-truncation-metadata"


class ServiceMetrics:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._values: dict[str, float] = {
            "requests": 0,
            "successes": 0,
            "failures": 0,
            "terminal_failures": 0,
            "retries": 0,
            "cache_hits": 0,
            "latency_seconds": 0,
            "response_bytes": 0,
        }

    def add(self, **values: float) -> None:
        with self._lock:
            for key, value in values.items():
                self._values[key] = self._values.get(key, 0) + value

    def snapshot(self) -> dict[str, float]:
        with self._lock:
            return dict(self._values)


class JsonFileCache:
    def __init__(self, root: str | Path):
        self.root = Path(root).expanduser()

    def _path(self, namespace: str, key: str) -> Path:
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
        return self.root / namespace / f"{digest}.json"

    def get(self, namespace: str, key: str) -> Any | None:
        path = self._path(namespace, key)
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def put(self, namespace: str, key: str, value: Any) -> None:
        path = self._path(namespace, key)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
        os.replace(temporary, path)


class RateLimiter:
    def __init__(self, requests_per_second: float):
        self.interval = 1.0 / requests_per_second
        self._lock = threading.Lock()
        self._next_allowed = 0.0

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            delay = max(self._next_allowed - now, 0.0)
            self._next_allowed = max(self._next_allowed, now) + self.interval
        if delay:
            time.sleep(delay)


class _SearchResultParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.results: list[dict[str, str]] = []
        self._current: dict[str, str] | None = None
        self._capture_title = False
        self._capture_snippet = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = dict(attrs)
        classes = set((values.get("class") or "").split())
        if tag == "a" and "result__a" in classes:
            self._current = {"title": "", "url": _decode_duckduckgo_url(values.get("href") or ""), "snippet": ""}
            self._capture_title = True
        elif self._current is not None and tag in {"a", "div"} and "result__snippet" in classes:
            self._capture_snippet = True

    def handle_endtag(self, tag: str) -> None:
        if tag == "a" and self._capture_title:
            self._capture_title = False
            if self._current and self._current["url"]:
                self.results.append(self._current)
            self._current = None
        if tag in {"a", "div"}:
            self._capture_snippet = False

    def handle_data(self, data: str) -> None:
        if self._capture_title and self._current is not None:
            self._current["title"] += data
        elif self._capture_snippet and self.results:
            self.results[-1]["snippet"] += data


class _ReadableTextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self._blocked_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style", "noscript", "svg"}:
            self._blocked_depth += 1
        elif not self._blocked_depth and tag in {"p", "br", "li", "h1", "h2", "h3", "tr"}:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript", "svg"} and self._blocked_depth:
            self._blocked_depth -= 1

    def handle_data(self, data: str) -> None:
        if not self._blocked_depth:
            stripped = " ".join(data.split())
            if stripped:
                self.parts.append(stripped)

    def text(self) -> str:
        return "\n".join(line.strip() for line in " ".join(self.parts).splitlines() if line.strip())


def _decode_duckduckgo_url(url: str) -> str:
    parsed = urllib.parse.urlparse(html.unescape(url))
    query = urllib.parse.parse_qs(parsed.query)
    if "uddg" in query:
        return query["uddg"][0]
    if url.startswith("//"):
        return "https:" + url
    return url


def _validate_public_url(url: str) -> None:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("only public http/https URLs are allowed")
    hostname = parsed.hostname.lower()
    if hostname == "localhost" or hostname.endswith(".local"):
        raise ValueError("local URLs are not allowed")
    try:
        address = ipaddress.ip_address(hostname)
    except ValueError:
        addresses = {
            item[4][0]
            for item in socket.getaddrinfo(hostname, parsed.port or 443, type=socket.SOCK_STREAM)
        }
        if not addresses:
            raise ValueError("URL hostname did not resolve")
        for item in addresses:
            if not ipaddress.ip_address(item).is_global:
                raise ValueError("private and loopback URLs are not allowed")
    else:
        if not address.is_global:
            raise ValueError("private and loopback URLs are not allowed")


class DeepDiveWebServices:
    def __init__(self, settings: SearchServiceConfig):
        self.settings = settings
        self.cache = JsonFileCache(settings.cache_dir)
        self.metrics = ServiceMetrics()
        self.rate_limiter = RateLimiter(settings.requests_per_second)
        self.semaphore = asyncio.Semaphore(settings.max_concurrency)

    async def search(self, arguments: dict[str, Any]) -> Any:
        query = str(arguments.get("query", "")).strip()
        if not query:
            raise ValueError("search query is empty")
        max_results = min(int(arguments.get("max_results", self.settings.max_results)), self.settings.max_results)
        key = json.dumps(
            {
                "cache_version": SEARCH_CACHE_VERSION,
                "provider": self.settings.provider,
                "query": query,
                "max_results": max_results,
            },
            sort_keys=True,
        )
        cached = self.cache.get("search", key)
        if cached is not None:
            self.metrics.add(cache_hits=1)
            return cached
        async with self.semaphore:
            result = await asyncio.to_thread(self._search_with_retries, query, max_results)
        self.cache.put("search", key, result)
        return result

    async def read_webpage(self, arguments: dict[str, Any]) -> dict[str, Any]:
        url = str(arguments.get("url", "")).strip()
        _validate_public_url(url)
        cached = self.cache.get(PAGE_CACHE_NAMESPACE, url)
        if cached is not None:
            self.metrics.add(cache_hits=1)
            return dict(cached)
        async with self.semaphore:
            result = await asyncio.to_thread(self._read_page_with_retries, url)
        self.cache.put(PAGE_CACHE_NAMESPACE, url, result)
        return result

    def _search_with_retries(self, query: str, max_results: int) -> dict[str, Any]:
        if self.settings.provider == "disabled":
            raise RuntimeError("search service is disabled because no provider is configured")
        last_error: Exception | None = None
        for attempt in range(self.settings.max_retries + 1):
            started = time.monotonic()
            self.metrics.add(requests=1)
            try:
                self.rate_limiter.wait()
                if self.settings.provider == "tavily":
                    results = self._search_tavily(query, max_results)
                    fallback_queries: list[str] = []
                else:
                    original_results = self._search_duckduckgo(query, max_results)
                    results = original_results
                    fallback_queries = []
                    if len(query.split()) > 36:
                        candidates = (
                            _tail_distinctive_search_query(query),
                            _distinctive_search_query(query),
                            _compact_search_query(query),
                            _focused_search_query(query),
                        )
                        for candidate in dict.fromkeys(candidates):
                            if candidate == query:
                                continue
                            fallback_queries.append(candidate)
                            candidate_results = self._search_duckduckgo(candidate, max_results)
                            if candidate_results:
                                results = _merge_search_results(
                                    candidate_results,
                                    original_results,
                                    max_results=max_results,
                                )
                                break
                results = self._enrich_truncated_doi_titles(results)
                self.metrics.add(successes=1, latency_seconds=time.monotonic() - started)
                return {
                    "provider": self.settings.provider,
                    "query": query,
                    "fallback_queries": fallback_queries,
                    "results": results,
                    "attempt_count": attempt + 1,
                }
            except Exception as exc:
                last_error = exc
                self.metrics.add(failures=1, latency_seconds=time.monotonic() - started)
                if attempt < self.settings.max_retries:
                    self.metrics.add(retries=1)
                    time.sleep(min(2**attempt, 8))
        self.metrics.add(terminal_failures=1)
        raise RuntimeError(f"search failed after retries: {last_error}") from last_error

    def _search_tavily(self, query: str, max_results: int) -> list[dict[str, Any]]:
        endpoint = self.settings.endpoint or "https://api.tavily.com/search"
        api_key = os.environ.get(self.settings.api_key_env, "")
        if not api_key:
            raise RuntimeError(f"Tavily requires environment variable {self.settings.api_key_env}")
        payload = json.dumps(
            {"api_key": api_key, "query": query, "max_results": max_results}
        ).encode("utf-8")
        request = urllib.request.Request(
            endpoint,
            data=payload,
            headers={"Content-Type": "application/json", "User-Agent": "RAO-Research-Agent/1.0"},
            method="POST",
        )
        response = self._open(request)
        parsed = json.loads(response["body"].decode("utf-8"))
        return [
            {
                "title": str(item.get("title", "")),
                "url": str(item.get("url", "")),
                "snippet": str(item.get("content", "")),
                "score": item.get("score"),
            }
            for item in parsed.get("results", [])[:max_results]
        ]

    def _search_duckduckgo(self, query: str, max_results: int) -> list[dict[str, Any]]:
        endpoint = self.settings.endpoint or "https://html.duckduckgo.com/html/"
        payload = urllib.parse.urlencode({"q": query}).encode("utf-8")
        request = urllib.request.Request(
            endpoint,
            data=payload,
            headers={
                "Content-Type": "application/x-www-form-urlencoded",
                "User-Agent": "Mozilla/5.0 RAO-Research-Agent/1.0",
            },
            method="POST",
        )
        response = self._open(request)
        parser = _SearchResultParser()
        parser.feed(response["body"].decode("utf-8", errors="replace"))
        return parser.results[:max_results]

    def _enrich_truncated_doi_titles(
        self,
        results: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        enriched: list[dict[str, Any]] = []
        titles_by_doi: dict[str, str | None] = {}
        for raw_item in results:
            item = dict(raw_item)
            title = str(item.get("title", "")).strip()
            doi = _extract_doi_from_url(str(item.get("url", "")))
            if _title_is_truncated(title) and doi:
                if doi not in titles_by_doi:
                    titles_by_doi[doi] = self._crossref_title(doi)
                complete_title = titles_by_doi[doi]
                if complete_title and len(complete_title) > len(title.rstrip(".… ")):
                    item["title"] = complete_title
                    item["title_source"] = "crossref_doi"
                    item["doi"] = doi
            enriched.append(item)
        return enriched

    def _crossref_title(self, doi: str) -> str | None:
        try:
            url = "https://api.crossref.org/works/" + urllib.parse.quote(
                doi,
                safe="/",
            )
            request = urllib.request.Request(
                url,
                headers={"User-Agent": "RAO-Research-Agent/1.0"},
            )
            response = self._open(request)
            payload = json.loads(response["body"].decode("utf-8"))
            titles = payload.get("message", {}).get("title", [])
            if titles and str(titles[0]).strip():
                return str(titles[0]).strip()
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
            return None
        return None

    def _read_page_with_retries(self, url: str) -> dict[str, Any]:
        last_error: Exception | None = None
        for attempt in range(self.settings.max_retries + 1):
            started = time.monotonic()
            self.metrics.add(requests=1)
            try:
                self.rate_limiter.wait()
                request = urllib.request.Request(
                    url,
                    headers={"User-Agent": "Mozilla/5.0 RAO-Research-Agent/1.0"},
                )
                response = self._open(request)
                parser = _ReadableTextParser()
                parser.feed(response["body"].decode("utf-8", errors="replace"))
                self.metrics.add(successes=1, latency_seconds=time.monotonic() - started)
                return {
                    "url": url,
                    "content": parser.text(),
                    "truncated": response["truncated"],
                    "bytes_read": response["bytes_read"],
                    "attempt_count": attempt + 1,
                }
            except Exception as exc:
                last_error = exc
                self.metrics.add(failures=1, latency_seconds=time.monotonic() - started)
                if attempt < self.settings.max_retries:
                    self.metrics.add(retries=1)
                    time.sleep(min(2**attempt, 8))
        self.metrics.add(terminal_failures=1)
        raise RuntimeError(f"webpage read failed after retries: {last_error}") from last_error

    def _open(self, request: urllib.request.Request) -> dict[str, Any]:
        with urllib.request.urlopen(request, timeout=self.settings.timeout_seconds) as response:
            raw_body = response.read(self.settings.max_content_bytes + 1)
        bytes_read = len(raw_body)
        truncated = bytes_read > self.settings.max_content_bytes
        body = raw_body[: self.settings.max_content_bytes]
        self.metrics.add(response_bytes=len(body))
        return {
            "body": body,
            "truncated": truncated,
            "bytes_read": bytes_read,
        }


def _compact_search_query(query: str) -> str:
    words = query.split()
    if len(words) <= 36:
        return query
    return " ".join(words[:18] + words[-18:])


def _title_is_truncated(title: str) -> bool:
    return title.rstrip().endswith(("...", "…"))


def _extract_doi_from_url(url: str) -> str | None:
    decoded = urllib.parse.unquote(url)
    match = re.search(
        r"(10\.\d{4,9}/[-._;()/:A-Z0-9]+)",
        decoded,
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    return match.group(1).rstrip(".,;")


_SEARCH_STOPWORDS = {
    "about",
    "and",
    "after",
    "another",
    "around",
    "been",
    "being",
    "between",
    "describing",
    "focused",
    "from",
    "characteristics",
    "effort",
    "international",
    "joint",
    "journal",
    "other",
    "paper",
    "principles",
    "published",
    "researchers",
    "specific",
    "their",
    "these",
    "this",
    "through",
    "types",
    "uniting",
    "within",
    "with",
    "work",
}


def _focused_search_query(query: str) -> str:
    words = re.findall(r"[A-Za-z0-9]+(?:-[A-Za-z0-9]+)?", query)
    focused = [
        word
        for word in words
        if len(word) > 2 and word.casefold() not in _SEARCH_STOPWORDS
    ]
    return " ".join(focused[-12:])


def _distinctive_search_query(query: str) -> str:
    focused = _focused_search_query(query).split()
    marked = []
    for content in re.findall(r"\(([^)]+)\)", query):
        marked.extend(re.findall(r"[A-Za-z0-9]+(?:-[A-Za-z0-9]+)?", content))
    marked.extend(re.findall(r"\b[A-Za-z0-9]+-[A-Za-z0-9-]+\b", query))
    prefix = list(dict.fromkeys(word for word in marked if len(word) > 2))
    used = {word.casefold() for word in prefix}
    remainder = [word for word in focused if word.casefold() not in used]
    return " ".join([*prefix, *remainder[-8:]])


def _tail_distinctive_search_query(query: str) -> str:
    parenthetical_groups = re.findall(r"\(([^)]+)\)", query)
    marked = (
        re.findall(
            r"[A-Za-z0-9]+(?:-[A-Za-z0-9]+)?",
            parenthetical_groups[-1],
        )
        if parenthetical_groups
        else []
    )
    marked.extend(re.findall(r"\b[A-Za-z0-9]+-[A-Za-z0-9-]+\b", query))
    prefix = list(dict.fromkeys(word for word in marked if len(word) > 2))
    used = {word.casefold() for word in prefix}
    focused = [
        word
        for word in _focused_search_query(query).split()
        if word.casefold() not in used
    ]
    return " ".join([*prefix, *focused[-6:]])


def _merge_search_results(
    preferred: list[dict[str, Any]],
    fallback: list[dict[str, Any]],
    *,
    max_results: int,
) -> list[dict[str, Any]]:
    merged = []
    seen = set()
    for item in [*preferred, *fallback]:
        key = (str(item.get("url", "")), str(item.get("title", "")).casefold())
        if key in seen:
            continue
        seen.add(key)
        merged.append(item)
        if len(merged) >= max_results:
            break
    return merged


def build_deepdive_tools(
    search_endpoint: str | None = None,
    api_key: str | None = None,
    *,
    settings: SearchServiceConfig | None = None,
) -> ToolRegistry:
    if settings is None:
        settings = SearchServiceConfig(
            provider="tavily" if search_endpoint else "disabled",
            endpoint=search_endpoint or "",
        )
        if api_key:
            os.environ.setdefault(settings.api_key_env, api_key)
    services = DeepDiveWebServices(settings)
    registry = ToolRegistry.with_default_tools()
    registry.register_tool(
        "search_web",
        services.search,
        "Search the public web and return titles, URLs, and snippets.",
        {"query": "string", "max_results": "integer"},
    )
    registry.register_tool(
        "view_webpage_content",
        services.read_webpage,
        "Read visible text from a public webpage.",
        {"url": "string"},
    )
    registry.service_metrics = services.metrics.snapshot  # type: ignore[attr-defined]
    return registry
