# Copyright (c) 2026 PGNS LLC
# SPDX-License-Identifier: Apache-2.0

"""CI regression gate: the 3-line quickstart must work after all A2A changes.

This test file ensures the minimal quickstart pattern — AgentServer(name,
description) + @on_task + test_client — works unmodified in local dev mode.
If any A2A change requires new constructor args or setup steps, these tests
must fail.
"""

from __future__ import annotations

from typing import Any

from starlette.testclient import TestClient

from pgns_agent import AgentServer, Task

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_echo_agent() -> AgentServer:
    """Create the minimal 3-line quickstart agent."""
    agent = AgentServer("echo", "echoes input")

    @agent.on_task
    async def handle(task: Task) -> dict[str, Any]:
        return {"echo": task.input}

    return agent


# ---------------------------------------------------------------------------
# TestThreeLineQuickstart
# ---------------------------------------------------------------------------


class TestThreeLineQuickstart:
    """Regression gate: bare constructor + bare decorator + local dev mode."""

    def test_minimal_constructor_and_handler(self) -> None:
        agent = _make_echo_agent()
        client = agent.test_client()
        resp = client.send_task({"text": "hello"})

        assert resp.status == "completed"
        assert resp.result == {"echo": {"text": "hello"}}
        assert resp.status_code == 200
        assert resp.error is None

    def test_agent_card_v1_shape(self) -> None:
        agent = _make_echo_agent()
        client = agent.test_client()
        card = client.build_agent_card()
        d = card.to_dict()

        # Core identity
        assert d["name"] == "echo"
        assert d["description"] == "echoes input"

        # v1.0: supportedInterfaces replaces top-level url
        assert "url" not in d
        interfaces = d["supportedInterfaces"]
        assert len(interfaces) == 1
        assert interfaces[0]["url"] == "http://localhost"
        assert interfaces[0]["protocolBinding"] == "HTTP+JSON"
        assert interfaces[0]["protocolVersion"] == "1.0"

        # v1.0 required fields
        assert d["capabilities"] == {
            "streaming": False,
            "pushNotifications": False,
            "extendedAgentCard": False,
        }
        assert d["defaultInputModes"] == ["application/json"]
        assert d["defaultOutputModes"] == ["application/json"]
        assert d["version"] == "0.0.0"

        # Bare handler excluded from skills
        assert d["skills"] == []

    def test_agent_card_http_endpoint(self) -> None:
        agent = _make_echo_agent()
        http = TestClient(agent.app())

        resp = http.get("/.well-known/agent.json")
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "application/json"

        body = resp.json()
        # v1.0 shape via HTTP
        assert "url" not in body
        assert "supportedInterfaces" in body
        assert body["supportedInterfaces"][0]["protocolBinding"] == "HTTP+JSON"
        assert "capabilities" in body
        assert "defaultInputModes" in body
        assert "defaultOutputModes" in body

    def test_post_message_send_a2a_envelope(self) -> None:
        agent = _make_echo_agent()
        client = agent.test_client()
        resp = client.send_a2a_message("hello")

        assert resp.status == "completed"
        assert resp.result == {"echo": "hello"}
        assert resp.status_code == 200
        assert resp.error is None

    def test_post_message_send_async(self) -> None:
        agent = _make_echo_agent()
        client = agent.test_client()
        resp = client.send_a2a_message("hello", blocking=False)

        assert resp.status == "submitted"
        assert resp.status_code == 202
