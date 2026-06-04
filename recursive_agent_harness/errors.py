"""Error types for the recursive agent harness."""


class HarnessError(Exception):
    """Base error for harness failures."""


class DepthLimitError(HarnessError):
    """Raised when a node tries to launch beyond max_depth."""


class NodeLimitError(HarnessError):
    """Raised when the global node budget is exhausted."""


class ChildLimitError(HarnessError):
    """Raised when a node exceeds its direct child budget."""


class StepLimitError(HarnessError):
    """Raised when a node exhausts its local step budget."""


class TimeoutError(HarnessError):
    """Raised when a run exceeds its configured timeout."""


class InvalidActionError(HarnessError):
    """Raised when a policy returns an invalid action."""


class ToolExecutionError(HarnessError):
    """Raised when a tool call fails unexpectedly."""


class PolicyError(HarnessError):
    """Raised when a policy cannot produce an action."""


class ActionParseError(PolicyError):
    """Raised when an LLM response cannot be parsed into an action."""


class CancelledExecutionError(HarnessError):
    """Raised when execution is cancelled."""
