"""Keep the test process offline unless a test explicitly opts into sockets."""

import socket

import pytest


@pytest.fixture(autouse=True)
def block_network(monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest) -> None:
    """Block socket connections and DNS, including loopback, for ordinary tests."""
    if request.node.get_closest_marker("network"):
        return

    def denied(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("network access is disabled in tests")

    for name in ("connect", "connect_ex", "sendto", "sendmsg", "bind"):
        if hasattr(socket.socket, name):
            monkeypatch.setattr(socket.socket, name, denied)
    for name in (
        "getaddrinfo",
        "gethostbyname",
        "gethostbyname_ex",
        "gethostbyaddr",
        "getnameinfo",
    ):
        monkeypatch.setattr(socket, name, denied)
