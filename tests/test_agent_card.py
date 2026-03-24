# Copyright (c) 2026 PGNS LLC
# SPDX-License-Identifier: Apache-2.0

"""Tests for Agent Card model and auto-generation from AgentServer."""

from __future__ import annotations

from starlette.applications import Starlette
from starlette.testclient import TestClient

from pgns_agent import (
    AgentCard,
    AgentCardAuthentication,
    AgentCardProvider,
    AgentCardSkill,
    AgentServer,
    Task,
)

# ---------------------------------------------------------------------------
# AgentCardSkill
# ---------------------------------------------------------------------------


class TestAgentCardSkill:
    def test_minimal(self) -> None:
        skill = AgentCardSkill(id="summarize")
        d = skill.to_dict()
        assert d == {"id": "summarize", "name": "summarize"}

    def test_with_name_and_description(self) -> None:
        skill = AgentCardSkill(id="summarize", name="Summarizer", description="Summarizes text")
        d = skill.to_dict()
        assert d == {"id": "summarize", "name": "Summarizer", "description": "Summarizes text"}

    def test_name_defaults_to_id(self) -> None:
        skill = AgentCardSkill(id="translate")
        assert skill.name == "translate"
        assert skill.to_dict()["name"] == "translate"


# ---------------------------------------------------------------------------
# AgentCardAuthentication
# ---------------------------------------------------------------------------


class TestAgentCardAuthentication:
    def test_empty(self) -> None:
        auth = AgentCardAuthentication()
        assert auth.to_dict() == {"schemes": []}

    def test_with_schemes(self) -> None:
        auth = AgentCardAuthentication(schemes=("bearer", "api_key"))
        d = auth.to_dict()
        assert d["schemes"] == ["bearer", "api_key"]

    def test_with_credentials(self) -> None:
        auth = AgentCardAuthentication(schemes=("bearer",), credentials="https://auth.example.com")
        d = auth.to_dict()
        assert d["credentials"] == "https://auth.example.com"

    def test_no_credentials_key_when_none(self) -> None:
        auth = AgentCardAuthentication(schemes=("bearer",))
        assert "credentials" not in auth.to_dict()


# ---------------------------------------------------------------------------
# AgentCardProvider
# ---------------------------------------------------------------------------


class TestAgentCardProvider:
    def test_minimal(self) -> None:
        provider = AgentCardProvider(organization="Acme Inc")
        d = provider.to_dict()
        assert d == {"organization": "Acme Inc"}
        assert "url" not in d

    def test_with_url(self) -> None:
        provider = AgentCardProvider(organization="Acme Inc", url="https://acme.com")
        d = provider.to_dict()
        assert d == {"organization": "Acme Inc", "url": "https://acme.com"}


# ---------------------------------------------------------------------------
# AgentCard
# ---------------------------------------------------------------------------


class TestAgentCard:
    def test_minimal(self) -> None:
        card = AgentCard(name="bot", description="A bot", url="http://localhost", version="1.0.0")
        d = card.to_dict()
        assert d["name"] == "bot"
        assert d["description"] == "A bot"
        assert d["url"] == "http://localhost"
        assert d["version"] == "1.0.0"
        assert d["skills"] == []
        assert "authentication" not in d
        assert "provider" not in d

    def test_with_all_fields(self) -> None:
        card = AgentCard(
            name="bot",
            description="A bot",
            url="http://localhost:8000",
            version="2.1.0",
            skills=(AgentCardSkill(id="summarize"),),
            authentication=AgentCardAuthentication(schemes=("bearer",)),
            provider=AgentCardProvider(organization="Acme"),
        )
        d = card.to_dict()
        assert len(d["skills"]) == 1
        assert d["skills"][0]["id"] == "summarize"
        assert d["authentication"]["schemes"] == ["bearer"]
        assert d["provider"]["organization"] == "Acme"


# ---------------------------------------------------------------------------
# AgentServer.build_agent_card()
# ---------------------------------------------------------------------------


class TestBuildAgentCard:
    def test_basic_card_from_constructor(self) -> None:
        agent = AgentServer("my-bot", "Does stuff", version="1.2.3")
        card = agent.build_agent_card(url="http://example.com")
        assert card.name == "my-bot"
        assert card.description == "Does stuff"
        assert card.url == "http://example.com"
        assert card.version == "1.2.3"
        assert card.skills == ()
        assert card.authentication is None
        assert card.provider is None

    def test_default_version(self) -> None:
        agent = AgentServer("a", "b")
        assert agent.version == "0.0.0"
        card = agent.build_agent_card()
        assert card.version == "0.0.0"

    def test_skills_from_named_handlers(self) -> None:
        agent = AgentServer("multi", "multi-skill")

        @agent.on_task("summarize")
        async def summarize(task: Task) -> dict[str, str]:
            return {}

        @agent.on_task("translate")
        async def translate(task: Task) -> dict[str, str]:
            return {}

        card = agent.build_agent_card()
        skill_ids = {s.id for s in card.skills}
        assert skill_ids == {"summarize", "translate"}

    def test_default_handler_excluded_from_skills(self) -> None:
        agent = AgentServer("a", "b")

        @agent.on_task
        async def handle(task: Task) -> dict[str, str]:
            return {}

        card = agent.build_agent_card()
        assert card.skills == ()

    def test_default_plus_named_only_shows_named(self) -> None:
        agent = AgentServer("a", "b")

        @agent.on_task
        async def fallback(task: Task) -> dict[str, str]:
            return {}

        @agent.on_task("special")
        async def special(task: Task) -> dict[str, str]:
            return {}

        card = agent.build_agent_card()
        assert len(card.skills) == 1
        assert card.skills[0].id == "special"

    def test_authentication_passed_through(self) -> None:
        auth = AgentCardAuthentication(schemes=("bearer",), credentials="https://auth.test")
        agent = AgentServer("a", "b", authentication=auth)
        card = agent.build_agent_card()
        assert card.authentication is auth

    def test_provider_passed_through(self) -> None:
        provider = AgentCardProvider(organization="Acme", url="https://acme.com")
        agent = AgentServer("a", "b", provider=provider)
        card = agent.build_agent_card()
        assert card.provider is provider

    def test_default_url(self) -> None:
        agent = AgentServer("a", "b")
        card = agent.build_agent_card()
        assert card.url == "http://localhost"


# ---------------------------------------------------------------------------
# AgentServer.agent_card_route() — HTTP endpoint
# ---------------------------------------------------------------------------


class TestAgentCardRoute:
    def test_serves_json_at_well_known_path(self) -> None:
        agent = AgentServer("test-agent", "For testing", version="0.1.0")

        @agent.on_task("echo")
        async def echo(task: Task) -> dict[str, str]:
            return {}

        app = Starlette(routes=[agent.agent_card_route(url="http://test:9000")])
        client = TestClient(app)

        resp = client.get("/.well-known/agent.json")
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "application/json"

        body = resp.json()
        assert body["name"] == "test-agent"
        assert body["description"] == "For testing"
        assert body["url"] == "http://test:9000"
        assert body["version"] == "0.1.0"
        assert len(body["skills"]) == 1
        assert body["skills"][0]["id"] == "echo"

    def test_cache_control_header(self) -> None:
        agent = AgentServer("a", "b")

        app = Starlette(routes=[agent.agent_card_route()])
        client = TestClient(app)

        resp = client.get("/.well-known/agent.json")
        assert resp.headers["cache-control"] == "public, max-age=3600"

    def test_only_get_allowed(self) -> None:
        agent = AgentServer("a", "b")

        app = Starlette(routes=[agent.agent_card_route()])
        client = TestClient(app)

        resp = client.post("/.well-known/agent.json")
        assert resp.status_code == 405

    def test_full_card_with_auth_and_provider(self) -> None:
        agent = AgentServer(
            "full-agent",
            "All fields",
            version="2.0.0",
            authentication=AgentCardAuthentication(
                schemes=("bearer",), credentials="https://auth.test"
            ),
            provider=AgentCardProvider(organization="TestCorp", url="https://testcorp.io"),
        )

        @agent.on_task("analyze")
        async def analyze(task: Task) -> dict[str, str]:
            return {}

        app = Starlette(routes=[agent.agent_card_route(url="https://agent.testcorp.io")])
        client = TestClient(app)

        body = client.get("/.well-known/agent.json").json()
        assert body["authentication"]["schemes"] == ["bearer"]
        assert body["authentication"]["credentials"] == "https://auth.test"
        assert body["provider"]["organization"] == "TestCorp"
        assert body["provider"]["url"] == "https://testcorp.io"
