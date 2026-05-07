from __future__ import annotations

from typing import Any, Protocol


class ProgressCallback(Protocol):
    def __call__( self, stage: str,
                        message: str,
                        fraction: float | None = None,
                        **extra: Any,
                ) -> None:
        ...


def noop_progress( stage: str,
                   message: str,
                   fraction: float | None = None,
                   **extra: Any, ) -> None:
    pass