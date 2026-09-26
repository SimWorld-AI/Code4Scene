"""Fixtures shared across test modules."""

from __future__ import annotations

import socket

import pytest

from code4scene.evaluation import report_schema

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
    for name in ("CODE4SCENE_VLM_BASE_URL", "CODE4SCENE_EMBED_BASE_URL"):
        monkeypatch.delenv(name, raising=False)


#: Every file that claims "conforms to VerifierReport" proves it by
#: constructing one from the packaged report contract.
@pytest.fixture(scope="module")
def infra():
    """The verifier report contract (``VerifierReport``, ``ReportStatus``)."""

    return report_schema
