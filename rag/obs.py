from __future__ import annotations

import logfire

from rag.config import Settings

_initialised = False


def setup_observability(cfg: Settings) -> None:
    """Configure Logfire — call once at app startup."""
    global _initialised
    if _initialised:
        return

    if cfg.LOGFIRE_TOKEN:
        logfire.configure(token=cfg.LOGFIRE_TOKEN)
        logfire.instrument_httpx()
        logfire.instrument_pydantic()
    else:
        logfire.configure(send_to_logfire=False)

    _initialised = True
