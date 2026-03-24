# Copyright (c) 2026 PGNS LLC
# SPDX-License-Identifier: Apache-2.0

"""Smoke test: package imports and exposes a version."""

from __future__ import annotations

import re

from pgns_agent import __version__


def test_version_is_string() -> None:
    assert isinstance(__version__, str)
    assert re.match(r"^\d+\.\d+\.\d+", __version__)
