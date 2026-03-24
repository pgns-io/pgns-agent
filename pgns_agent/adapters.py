# Copyright (c) 2026 PGNS LLC
# SPDX-License-Identifier: Apache-2.0

"""Convenience re-exports for framework adapter packages.

Install the adapter package for your framework::

    pip install pgns-agent-langchain   # LangChain / LangGraph

Then import the adapter from here::

    from pgns_agent.adapters import LangChainAdapter
    agent.use(LangChainAdapter(my_chain))
"""

from __future__ import annotations

import importlib
from typing import Any

_ADAPTERS: dict[str, tuple[str, str]] = {
    "LangChainAdapter": ("pgns_agent_langchain", "LangChainAdapter"),
    "LangChainStreamAdapter": ("pgns_agent_langchain", "LangChainStreamAdapter"),
}


def __getattr__(name: str) -> Any:
    if name in _ADAPTERS:
        module_name, attr = _ADAPTERS[name]
        try:
            mod = importlib.import_module(module_name)
        except ModuleNotFoundError as exc:
            raise ImportError(
                f"{name} requires the '{module_name.replace('_', '-')}' package. "
                f"Install it with: pip install {module_name.replace('_', '-')}"
            ) from exc
        cls = getattr(mod, attr)
        globals()[name] = cls
        return cls
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return list(_ADAPTERS.keys())
