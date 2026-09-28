"""The autouse guard really blocks real network access."""

import socket

import httpx
import pytest

from conftest import NetworkBlockedError


def test_httpx_real_transport_blocked():
    with httpx.Client() as client, pytest.raises(NetworkBlockedError):
        client.get("https://api.crossref.org/works")


def test_socket_blocked():
    with pytest.raises(NetworkBlockedError):
        socket.create_connection(("example.org", 443))
