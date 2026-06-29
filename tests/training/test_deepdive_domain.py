import pytest

from recursive_agent_training.config import SearchServiceConfig
from recursive_agent_training.domains.deepdive import (
    DeepDiveWebServices,
    SEARCH_CACHE_VERSION,
    _distinctive_search_query,
    _extract_doi_from_url,
    _focused_search_query,
    _merge_search_results,
    _tail_distinctive_search_query,
    build_deepdive_tools,
)


@pytest.mark.asyncio
async def test_deepdive_search_requires_configuration():
    tools = build_deepdive_tools()
    result = await tools.call("search_web", {"query": "test"})
    assert result.ok is False
    assert "configured" in (result.error or "")


@pytest.mark.asyncio
async def test_duckduckgo_search_is_normalized_and_cached(tmp_path, monkeypatch):
    html = b"""
    <html><body>
      <a rel="nofollow" class="result__a"
         href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fpaper">Paper Title</a>
      <a class="result__snippet">Useful snippet</a>
    </body></html>
    """
    calls = 0

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self, size):
            return html

    def fake_urlopen(request, timeout):
        nonlocal calls
        calls += 1
        return Response()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    settings = SearchServiceConfig(
        provider="duckduckgo",
        cache_dir=str(tmp_path),
        requests_per_second=1000,
    )
    tools = build_deepdive_tools(settings=settings)
    first = await tools.call("search_web", {"query": "test paper"})
    second = await tools.call("search_web", {"query": "test paper"})
    assert first.ok is True
    assert first.result["results"][0]["title"].strip() == "Paper Title"
    assert first.result["results"][0]["url"] == "https://example.com/paper"
    assert second.result == first.result
    assert calls == 1
    assert tools.service_metrics()["cache_hits"] == 1
    assert tools.service_metrics()["terminal_failures"] == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("body", "limit", "expected_content", "expected_truncated", "expected_bytes_read"),
    [
        (b"complete page", 32, "complete page", False, 13),
        (b"abcdefghijklmnop", 10, "abcdefghij", True, 11),
    ],
)
async def test_webpage_read_exposes_truncation_metadata(
    tmp_path,
    monkeypatch,
    body,
    limit,
    expected_content,
    expected_truncated,
    expected_bytes_read,
):
    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self, size):
            return body[:size]

    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda request, timeout: Response(),
    )
    service = DeepDiveWebServices(
        SearchServiceConfig(
            provider="duckduckgo",
            cache_dir=str(tmp_path),
            max_content_bytes=limit,
            max_retries=0,
            requests_per_second=1000,
        )
    )

    result = await service.read_webpage({"url": "https://93.184.216.34/page"})

    assert result["content"] == expected_content
    assert result["truncated"] is expected_truncated
    assert result["bytes_read"] == expected_bytes_read


def test_focused_query_keeps_distinctive_tail_terms():
    query = (
        "2012 paper published in an international journal with many biographical "
        "details and their joint effort with energy and momentum conservation "
        "principles, surface acoustic phenomena, and particle-like characteristics (grains)"
    )
    focused = _focused_search_query(query)
    assert "grains" in focused
    assert "surface acoustic phenomena" in focused
    assert len(focused.split()) <= 12
    distinctive = _distinctive_search_query(query)
    assert distinctive.startswith("grains particle-like")
    assert "surface acoustic phenomena" in distinctive
    tail_distinctive = _tail_distinctive_search_query(query)
    assert tail_distinctive == (
        "grains particle-like energy momentum conservation surface acoustic phenomena"
    )
    assert SEARCH_CACHE_VERSION == "deepdive-search-v5-crossref-doi-title"


def test_search_result_merge_prioritizes_focused_results():
    merged = _merge_search_results(
        [{"title": "Target", "url": "https://example.com/target"}],
        [
            {"title": "Noise", "url": "https://example.com/noise"},
            {"title": "Target", "url": "https://example.com/target"},
        ],
        max_results=2,
    )
    assert [item["title"] for item in merged] == ["Target", "Noise"]


def test_truncated_doi_title_is_enriched_from_crossref(tmp_path, monkeypatch):
    settings = SearchServiceConfig(
        provider="duckduckgo",
        cache_dir=str(tmp_path),
        requests_per_second=1000,
    )
    service = DeepDiveWebServices(settings)
    monkeypatch.setattr(
        service,
        "_crossref_title",
        lambda doi: (
            "Visualization and Visual Analytics Approaches for Image and "
            "Video Datasets: A Survey"
        ),
    )

    enriched = service._enrich_truncated_doi_titles(
        [
            {
                "title": (
                    "Visualization and Visual Analytics Approaches for Image "
                    "and Video ..."
                ),
                "url": "https://dl.acm.org/doi/full/10.1145/3576935",
                "snippet": "",
            }
        ]
    )

    assert enriched[0]["title"].endswith("Datasets: A Survey")
    assert enriched[0]["title_source"] == "crossref_doi"
    assert enriched[0]["doi"] == "10.1145/3576935"
    assert _extract_doi_from_url(enriched[0]["url"]) == "10.1145/3576935"
