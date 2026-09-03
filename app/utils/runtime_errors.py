"""Stable operational failures shared by request orchestration layers.

These errors describe the application's own runtime contracts rather than an
upstream provider response.  Keeping them typed lets API routes distinguish a
busy local model, an expired foreground budget, and an unavailable index
without exposing implementation details to users.
"""

from __future__ import annotations


class GenerationQueueTimeoutError(RuntimeError):
    """Foreground work could not obtain a generation slot in time."""

    def __init__(self, timeout_seconds: float) -> None:
        self.timeout_seconds = timeout_seconds
        super().__init__(f"generation queue wait exceeded {timeout_seconds:.1f}s")


class ConversationTurnQueueTimeoutError(RuntimeError):
    """A second turn for one conversation waited too long for ordering."""

    def __init__(self, timeout_seconds: float) -> None:
        self.timeout_seconds = timeout_seconds
        super().__init__(f"conversation turn queue wait exceeded {timeout_seconds:.1f}s")


class ChatTurnDeadlineExceededError(RuntimeError):
    """One foreground chat turn exceeded its end-to-end wall-clock budget."""

    def __init__(self, timeout_seconds: float) -> None:
        self.timeout_seconds = timeout_seconds
        super().__init__(f"chat turn exceeded {timeout_seconds:.1f}s")


class IndexUnavailableError(RuntimeError):
    """Metadata points to a physical index collection which is absent."""

    def __init__(self, collection_name: str) -> None:
        self.collection_name = collection_name
        super().__init__(
            "知识库索引暂不可用，系统将尝试恢复；如持续存在，请在文档页重新建立索引。"
        )


class AnswerVerificationUnavailableError(RuntimeError):
    """The semantic answer verifier did not return a usable decision."""

    def __init__(self) -> None:
        super().__init__("answer semantic verification unavailable")


class AnswerVerificationFailedError(RuntimeError):
    """The answer remained unsupported after its bounded repair budget."""

    def __init__(self) -> None:
        super().__init__("answer did not pass semantic evidence verification")
