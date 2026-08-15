from __future__ import annotations

import tomllib
from pathlib import Path

import pytest
from h2.config import H2Configuration
from h2.connection import H2Connection
from h2.errors import ErrorCodes
from h2.events import RequestReceived
from h2.exceptions import ProtocolError

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def request_bytes(host_headers: list[tuple[str, str]]) -> bytes:
    client = H2Connection(
        H2Configuration(client_side=True, validate_outbound_headers=False)
    )
    client.initiate_connection()
    preface = client.data_to_send()
    client.send_headers(
        1,
        [
            (":method", "GET"),
            (":scheme", "https"),
            (":path", "/"),
            *host_headers,
        ],
        end_stream=True,
    )
    return preface + client.data_to_send()


def server_connection() -> H2Connection:
    server = H2Connection(H2Configuration(client_side=False, header_encoding="utf-8"))
    server.initiate_connection()
    server.data_to_send()
    return server


def test_duplicate_host_headers_are_rejected_before_application_delivery() -> None:
    server = server_connection()
    with pytest.raises(ProtocolError):
        server.receive_data(
            request_bytes([("host", "one.test"), ("host", "two.test")])
        )
    assert server.state_machine.state.name == "CLOSED"
    assert ErrorCodes.PROTOCOL_ERROR.value.to_bytes(4, "big") in server.data_to_send()


def test_single_host_header_still_reaches_the_application() -> None:
    events = server_connection().receive_data(request_bytes([("host", "one.test")]))
    request = next(event for event in events if isinstance(event, RequestReceived))
    assert ("host", "one.test") in request.headers


def test_lock_uses_patched_h2_and_registry_only_sources() -> None:
    lock = tomllib.loads((REPOSITORY_ROOT / "uv.lock").read_text("utf-8"))
    h2 = next(package for package in lock["package"] if package["name"] == "h2")
    assert tuple(int(part) for part in h2["version"].split(".")) >= (4, 4, 1)

    for package in lock["package"]:
        source = package.get("source", {})
        assert "git" not in source, package["name"]
        assert "url" not in source, package["name"]
        assert "path" not in source, package["name"]
        if registry := source.get("registry"):
            assert registry == "https://pypi.org/simple"
