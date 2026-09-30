"""Fixtures shared across test modules."""

from __future__ import annotations

import os
import socket

import pytest

#: Judge, embedding and other scoring settings a user may have exported (see
#: the README) must not reach the tests. Some modules read them at import, so
#: they are cleared here, before anything from code4scene is imported, and
#: again before every test.
_USER_SETTINGS = ("CODE4SCENE_", "OPENAI_API_KEY")


def _user_settings() -> list[str]:
    return [name for name in os.environ if name.startswith(_USER_SETTINGS)]


for _name in _user_settings():
    del os.environ[_name]

from code4scene.evaluation import report_schema  # noqa: E402

_ALLOWED_HOSTS = {"127.0.0.1", "::1", "localhost"}
_real_connect = socket.socket.connect


def _guarded_connect(self, address):
    host = address[0] if isinstance(address, tuple) else address
    if isinstance(address, tuple) and str(host) not in _ALLOWED_HOSTS:
        raise OSError(f"tests run offline; refused a connection to {host!r}")
    return _real_connect(self, address)


@pytest.fixture(autouse=True)
def _offline(monkeypatch):
    """Every test runs offline: any non-loopback connection fails loudly."""

    monkeypatch.setattr(socket.socket, "connect", _guarded_connect)
    for name in _user_settings():
        monkeypatch.delenv(name, raising=False)


#: Every file that claims "conforms to VerifierReport" proves it by
#: constructing one from the packaged report contract.
@pytest.fixture(scope="module")
def infra():
    """The verifier report contract (``VerifierReport``, ``ReportStatus``)."""

    return report_schema
