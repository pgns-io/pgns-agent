# Copyright (c) 2026 PGNS LLC
# SPDX-License-Identifier: Apache-2.0

"""Tests for Agent Card model and auto-generation from AgentServer."""

from __future__ import annotations

import warnings

import pytest
from pydantic import BaseModel
from starlette.applications import Starlette
from starlette.testclient import TestClient

from pgns_agent import (
    AgentCapabilities,
    AgentCard,
    AgentCardAuthentication,
    AgentCardProvider,
    AgentCardSecurityScheme,
    AgentCardSkill,
    AgentServer,
    Task,
)


class _SampleInput(BaseModel):
    text: str
    max_length: int = 100


# ---------------------------------------------------------------------------
# AgentCardSkill
# ---------------------------------------------------------------------------


class TestAgentCardSkill:
    def test_minimal(self) -> None:
        skill = AgentCardSkill(id="summarize")
        d = skill.to_dict()
        assert d == {"id": "summarize", "name": "summarize", "description": "", "tags": []}

    def test_with_name_and_description(self) -> None:
        skill = AgentCardSkill(id="summarize", name="Summarizer", description="Summarizes text")
        d = skill.to_dict()
        assert d == {
            "id": "summarize",
            "name": "Summarizer",
            "description": "Summarizes text",
            "tags": [],
        }

    def test_name_defaults_to_id(self) -> None:
        skill = AgentCardSkill(id="translate")
        assert skill.name == "translate"
        assert skill.to_dict()["name"] == "translate"

    def test_description_defaults_to_none(self) -> None:
        skill = AgentCardSkill(id="x")
        assert skill.description is None
        assert skill.to_dict()["description"] == ""

    def test_tags_default_to_empty(self) -> None:
        skill = AgentCardSkill(id="x")
        assert skill.tags == ()
        assert skill.to_dict()["tags"] == []

    def test_tags_with_values(self) -> None:
        skill = AgentCardSkill(id="x", tags=("nlp", "summarization"))
        assert skill.to_dict()["tags"] == ["nlp", "summarization"]


# ---------------------------------------------------------------------------
# AgentCapabilities
# ---------------------------------------------------------------------------


class TestAgentCapabilities:
    def test_defaults_all_false(self) -> None:
        caps = AgentCapabilities()
        assert caps.streaming is False
        assert caps.push_notifications is False
        assert caps.extended_agent_card is False

    def test_custom_values(self) -> None:
        caps = AgentCapabilities(streaming=True, push_notifications=True)
        assert caps.streaming is True
        assert caps.push_notifications is True
        assert caps.extended_agent_card is False

    def test_to_dict(self) -> None:
        caps = AgentCapabilities(streaming=True)
        d = caps.to_dict()
        assert d == {
            "streaming": True,
            "pushNotifications": False,
            "extendedAgentCard": False,
        }


# ---------------------------------------------------------------------------
# AgentCardSecurityScheme
# ---------------------------------------------------------------------------


class TestAgentCardSecurityScheme:
    def test_to_dict_without_credentials(self) -> None:
        scheme = AgentCardSecurityScheme(scheme="bearer")
        assert scheme.to_dict() == {"scheme": "bearer"}

    def test_to_dict_with_credentials(self) -> None:
        scheme = AgentCardSecurityScheme(scheme="bearer", credentials="https://auth.example.com")
        assert scheme.to_dict() == {
            "scheme": "bearer",
            "credentials": "https://auth.example.com",
        }

    def test_with_input_schema(self) -> None:
        schema = {"type": "object", "properties": {"text": {"type": "string"}}}
        skill = AgentCardSkill(id="summarize", input_schema=schema)
        d = skill.to_dict()
        assert d["inputSchema"] == schema

    def test_with_output_schema(self) -> None:
        schema = {"type": "object", "properties": {"result": {"type": "string"}}}
        skill = AgentCardSkill(id="summarize", output_schema=schema)
        d = skill.to_dict()
        assert d["outputSchema"] == schema

    def test_with_both_schemas(self) -> None:
        in_schema = {"type": "object", "properties": {"text": {"type": "string"}}}
        out_schema = {"type": "object", "properties": {"result": {"type": "string"}}}
        skill = AgentCardSkill(id="summarize", input_schema=in_schema, output_schema=out_schema)
        d = skill.to_dict()
        assert d["inputSchema"] == in_schema
        assert d["outputSchema"] == out_schema

    def test_schemas_omitted_when_none(self) -> None:
        skill = AgentCardSkill(id="summarize")
        d = skill.to_dict()
        assert "inputSchema" not in d
        assert "outputSchema" not in d


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
        assert "url" not in d
        assert d["supportedInterfaces"] == [
            {"url": "http://localhost", "protocolBinding": "HTTP+JSON", "protocolVersion": "1.0"}
        ]
        assert d["version"] == "1.0.0"
        assert d["skills"] == []
        assert d["capabilities"] == {
            "streaming": False,
            "pushNotifications": False,
            "extendedAgentCard": False,
        }
        assert d["defaultInputModes"] == ["application/json"]
        assert d["defaultOutputModes"] == ["application/json"]
        assert "securitySchemes" not in d
        assert "authentication" not in d
        assert "provider" not in d

    def test_with_all_fields(self) -> None:
        card = AgentCard(
            name="bot",
            description="A bot",
            url="http://localhost:8000",
            version="2.1.0",
            skills=(AgentCardSkill(id="summarize"),),
            capabilities=AgentCapabilities(streaming=True),
            default_input_modes=("text/plain",),
            default_output_modes=("text/plain", "application/json"),
            security_schemes=(AgentCardSecurityScheme(scheme="bearer"),),
            authentication=AgentCardAuthentication(schemes=("bearer",)),
            provider=AgentCardProvider(organization="Acme"),
        )
        d = card.to_dict()
        assert "url" not in d
        assert d["supportedInterfaces"] == [
            {
                "url": "http://localhost:8000",
                "protocolBinding": "HTTP+JSON",
                "protocolVersion": "1.0",
            }
        ]
        assert len(d["skills"]) == 1
        assert d["skills"][0]["id"] == "summarize"
        assert d["capabilities"]["streaming"] is True
        assert d["defaultInputModes"] == ["text/plain"]
        assert d["defaultOutputModes"] == ["text/plain", "application/json"]
        assert d["securitySchemes"] == [{"scheme": "bearer"}]
        assert d["authentication"]["schemes"] == ["bearer"]
        assert d["provider"]["organization"] == "Acme"

    def test_security_schemes_omitted_when_empty(self) -> None:
        card = AgentCard(name="bot", description="A bot", url="http://localhost", version="1.0.0")
        assert "securitySchemes" not in card.to_dict()


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
        assert card.capabilities == AgentCapabilities()
        assert card.default_input_modes == ("application/json",)
        assert card.default_output_modes == ("application/json",)
        assert card.security_schemes == ()
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
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
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

    def test_capabilities_passed_through(self) -> None:
        caps = AgentCapabilities(streaming=True)
        agent = AgentServer("a", "b", capabilities=caps)
        card = agent.build_agent_card()
        assert card.capabilities is caps

    def test_default_modes_passed_through(self) -> None:
        agent = AgentServer(
            "a",
            "b",
            default_input_modes=("text/plain",),
            default_output_modes=("text/plain",),
        )
        card = agent.build_agent_card()
        assert card.default_input_modes == ("text/plain",)
        assert card.default_output_modes == ("text/plain",)

    def test_security_schemes_passed_through(self) -> None:
        schemes = (AgentCardSecurityScheme(scheme="bearer"),)
        agent = AgentServer("a", "b", security_schemes=schemes)
        card = agent.build_agent_card()
        assert card.security_schemes == schemes

    def test_authentication_deprecation_warning(self) -> None:
        auth = AgentCardAuthentication(schemes=("bearer",), credentials="https://auth.test")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            AgentServer("a", "b", authentication=auth)
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "security_schemes" in str(w[0].message)

    def test_authentication_auto_converts_to_security_schemes(self) -> None:
        auth = AgentCardAuthentication(schemes=("bearer", "api_key"), credentials="https://creds")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            agent = AgentServer("a", "b", authentication=auth)
        card = agent.build_agent_card()
        assert len(card.security_schemes) == 2
        assert card.security_schemes[0].scheme == "bearer"
        assert card.security_schemes[0].credentials == "https://creds"
        assert card.security_schemes[1].scheme == "api_key"

    def test_explicit_security_schemes_wins_over_authentication(self) -> None:
        auth = AgentCardAuthentication(schemes=("bearer",))
        schemes = (AgentCardSecurityScheme(scheme="oauth2"),)
        # No deprecation warning when security_schemes is explicitly set
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            agent = AgentServer("a", "b", authentication=auth, security_schemes=schemes)
            dep_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert len(dep_warnings) == 0
        card = agent.build_agent_card()
        assert card.security_schemes == schemes

    def test_schema_from_pydantic_model(self) -> None:
        agent = AgentServer("a", "b")

        @agent.on_task("summarize", schema=_SampleInput)
        async def summarize(task: Task) -> dict[str, str]:
            return {}

        card = agent.build_agent_card()
        assert len(card.skills) == 1
        assert card.skills[0].input_schema == _SampleInput.model_json_schema()

    def test_schema_appears_in_card_json(self) -> None:
        agent = AgentServer("a", "b")

        @agent.on_task("summarize", schema=_SampleInput)
        async def summarize(task: Task) -> dict[str, str]:
            return {}

        d = agent.build_agent_card().to_dict()
        skill = d["skills"][0]
        assert "inputSchema" in skill
        assert skill["inputSchema"]["properties"]["text"]["type"] == "string"

    def test_no_schema_no_key(self) -> None:
        agent = AgentServer("a", "b")

        @agent.on_task("summarize")
        async def summarize(task: Task) -> dict[str, str]:
            return {}

        d = agent.build_agent_card().to_dict()
        assert "inputSchema" not in d["skills"][0]
        assert "outputSchema" not in d["skills"][0]

    def test_schema_without_pydantic_raises(self) -> None:
        agent = AgentServer("a", "b")

        class PlainClass:
            pass

        with pytest.raises(TypeError, match="Unsupported schema type"):

            @agent.on_task("summarize", schema=PlainClass)
            async def summarize(task: Task) -> dict[str, str]:
                return {}


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
        assert "url" not in body
        assert body["supportedInterfaces"] == [
            {"url": "http://test:9000", "protocolBinding": "HTTP+JSON", "protocolVersion": "1.0"}
        ]
        assert body["version"] == "0.1.0"
        assert len(body["skills"]) == 1
        assert body["skills"][0]["id"] == "echo"
        # v1.0 required fields present
        assert "capabilities" in body
        assert "defaultInputModes" in body
        assert "defaultOutputModes" in body

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
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
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
        assert "url" not in body
        assert body["supportedInterfaces"] == [
            {
                "url": "https://agent.testcorp.io",
                "protocolBinding": "HTTP+JSON",
                "protocolVersion": "1.0",
            }
        ]
        assert body["authentication"]["schemes"] == ["bearer"]
        assert body["authentication"]["credentials"] == "https://auth.test"
        assert body["provider"]["organization"] == "TestCorp"
        assert body["provider"]["url"] == "https://testcorp.io"

    def test_v1_fields_in_http_response(self) -> None:
        agent = AgentServer(
            "v1-agent",
            "v1.0 compliant",
            capabilities=AgentCapabilities(streaming=True),
        )

        app = Starlette(routes=[agent.agent_card_route()])
        client = TestClient(app)

        body = client.get("/.well-known/agent.json").json()
        assert body["capabilities"]["streaming"] is True
        assert body["defaultInputModes"] == ["application/json"]
        assert body["defaultOutputModes"] == ["application/json"]

    def test_skill_schemas_served_via_http(self) -> None:
        agent = AgentServer("schema-agent", "Agent with schemas", version="1.0.0")

        @agent.on_task("summarize", schema=_SampleInput)
        async def summarize(task: Task) -> dict[str, str]:
            return {}

        app = Starlette(routes=[agent.agent_card_route(url="http://test:9000")])
        client = TestClient(app)

        body = client.get("/.well-known/agent.json").json()
        skill = body["skills"][0]
        assert skill["id"] == "summarize"
        assert "inputSchema" in skill
        assert skill["inputSchema"]["properties"]["text"]["type"] == "string"
        assert skill["inputSchema"]["properties"]["max_length"]["default"] == 100
