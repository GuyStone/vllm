# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for --host server socket binding across address families."""

import errno
import socket
import sys
from http import HTTPStatus

import pytest
import requests

from vllm.utils.network_utils import create_server_sockets, get_open_port

from ...utils import RemoteOpenAIServer

MODEL_NAME = "hmellor/tiny-random-LlamaForCausalLM"


def _can_bind(family: socket.AddressFamily, addr: str) -> bool:
    """Probe support for a family by binding its loopback address.

    ``getaddrinfo`` cannot be used here: with ``AI_PASSIVE`` it reports IPv6
    as available even on hosts with IPv6 disabled.
    """
    try:
        with socket.socket(family, socket.SOCK_STREAM) as sock:
            sock.bind((addr, 0))
        return True
    except OSError:
        return False


HAS_IPV4 = _can_bind(socket.AF_INET, "127.0.0.1")
HAS_IPV6 = _can_bind(socket.AF_INET6, "::1")


def _localhost_families() -> set[socket.AddressFamily]:
    try:
        infos = socket.getaddrinfo("localhost", None, type=socket.SOCK_STREAM)
    except socket.gaierror:
        return set()
    return {info[0] for info in infos}


_LOCALHOST_FAMILIES = _localhost_families()
LOCALHOST_ADDRS = [
    addr
    for family, addr in [
        (socket.AF_INET, "127.0.0.1"),
        (socket.AF_INET6, "::1"),
    ]
    if family in _LOCALHOST_FAMILIES
]


def _bindv6only() -> bool:
    """Kernel default for IPV6_V6ONLY; 1 disables dual-stack '::' sockets."""
    try:
        with open("/proc/sys/net/ipv6/bindv6only") as f:
            return f.read().strip() == "1"
    except OSError:
        return False


BINDV6ONLY = _bindv6only()


def _family_available(addr: str) -> bool:
    return HAS_IPV6 if addr == "::1" else HAS_IPV4


def _close_all(sockets: list[socket.socket]) -> None:
    for sock in sockets:
        sock.close()


@pytest.fixture(scope="function")
def server(request: pytest.FixtureRequest):
    args = [
        # use half precision for speed and memory savings in CI environment
        "--dtype",
        "bfloat16",
        "--max-model-len",
        "2048",
        "--enforce-eager",
        "--max-num-seqs",
        "128",
        # binding behavior does not depend on KV cache size; keep the
        # footprint small so the test runs on busy GPUs
        "--gpu-memory-utilization",
        "0.3",
        *request.param,
    ]

    with RemoteOpenAIServer(MODEL_NAME, args) as remote_server:
        yield remote_server


@pytest.mark.skipif(
    sys.platform != "linux", reason="relies on Linux wildcard-connect semantics"
)
@pytest.mark.parametrize(
    ("server", "expected_addrs", "unexpected_addrs"),
    [
        pytest.param(
            [],
            ["127.0.0.1", "::1"],
            [],
            id="default",
            # the fixture's readiness poll targets 127.0.0.1
            marks=pytest.mark.skipif(not HAS_IPV4, reason="requires IPv4"),
        ),
        pytest.param(
            ["--host=0.0.0.0"],
            ["127.0.0.1"],
            ["::1"],
            id="ipv4-wildcard",
            marks=pytest.mark.skipif(not HAS_IPV4, reason="requires IPv4"),
        ),
        pytest.param(
            ["--host=127.0.0.1"],
            ["127.0.0.1"],
            ["::1"],
            id="ipv4-loopback",
            marks=pytest.mark.skipif(not HAS_IPV4, reason="requires IPv4"),
        ),
        pytest.param(
            ["--host=::1"],
            ["::1"],
            ["127.0.0.1"],
            id="ipv6-loopback",
            marks=pytest.mark.skipif(not HAS_IPV6, reason="requires IPv6"),
        ),
        pytest.param(
            ["--host=::"],
            ["127.0.0.1", "::1"],
            [],
            id="ipv6-wildcard",
            marks=[
                pytest.mark.skipif(not HAS_IPV6, reason="requires IPv6"),
                pytest.mark.skipif(
                    BINDV6ONLY,
                    reason="kernel disables dual-stack '::' sockets",
                ),
            ],
        ),
        pytest.param(
            ["--host=localhost"],
            LOCALHOST_ADDRS,
            [],
            id="localhost",
            marks=pytest.mark.skipif(
                not LOCALHOST_ADDRS, reason="localhost does not resolve"
            ),
        ),
    ],
    indirect=["server"],
)
def test_bind_ipv4_ipv6(
    server: RemoteOpenAIServer,
    expected_addrs: list[str],
    unexpected_addrs: list[str],
):
    """Each --host value serves exactly the address families in the contract:
    unset and localhost bind both families, wildcards and literals keep their
    single-family (or kernel dual-stack) behavior."""
    # Probing an address whose family this machine lacks is meaningless in
    # both directions; drop it rather than let the request fail on name
    # resolution instead of connection refusal.
    expected = [addr for addr in expected_addrs if _family_available(addr)]
    unexpected = [addr for addr in unexpected_addrs if _family_available(addr)]

    for addr in expected:
        response = requests.get(server.url_for_host(addr, "health"), timeout=5)
        assert response.status_code == HTTPStatus.OK, addr

    for addr in unexpected:
        with pytest.raises(requests.ConnectionError):
            requests.get(server.url_for_host(addr, "health"), timeout=5)


def _create_sockets_on_open_port(
    host: str | None,
) -> tuple[int, list[socket.socket]]:
    """Bind a fresh port, retrying if another process grabs it first."""
    last_error: OSError | None = None
    for _ in range(5):
        port = get_open_port()
        try:
            return port, create_server_sockets((host, port), reuse_port=False)
        except OSError as exc:
            if exc.errno != errno.EADDRINUSE:
                raise
            last_error = exc
    raise AssertionError("could not find a free port") from last_error


@pytest.mark.parametrize(
    ("host", "family"),
    [
        pytest.param(
            "127.0.0.1",
            socket.AF_INET,
            id="ipv4",
            marks=pytest.mark.skipif(not HAS_IPV4, reason="requires IPv4"),
        ),
        pytest.param(
            "::1",
            socket.AF_INET6,
            id="ipv6",
            marks=pytest.mark.skipif(not HAS_IPV6, reason="requires IPv6"),
        ),
    ],
)
def test_literal_host_binds_single_socket(host: str, family: socket.AddressFamily):
    """Literal addresses take the single-socket fast path, no resolver."""
    sockets = create_server_sockets((host, 0), reuse_port=False)
    try:
        assert len(sockets) == 1
        assert sockets[0].family == family
    finally:
        _close_all(sockets)


@pytest.mark.skipif(not HAS_IPV6, reason="requires IPv6")
@pytest.mark.skipif(
    not hasattr(socket, "IPPROTO_IPV6"), reason="platform lacks IPPROTO_IPV6"
)
def test_literal_v6_wildcard_keeps_kernel_v6only_default():
    """A literal '::' must not force IPV6_V6ONLY: the kernel default decides
    whether the socket is dual-stack."""
    with socket.socket(socket.AF_INET6, socket.SOCK_STREAM) as probe:
        kernel_default = probe.getsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY)
    sockets = create_server_sockets(("::", 0), reuse_port=False)
    try:
        assert len(sockets) == 1
        assert (
            sockets[0].getsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY)
            == kernel_default
        )
    finally:
        _close_all(sockets)


def test_unspecified_host_binds_available_families():
    """host=None binds one socket per available family, all on one port,
    with resolved AF_INET6 wildcards isolated from the IPv4 bind."""
    port, sockets = _create_sockets_on_open_port(None)
    try:
        families = {sock.family for sock in sockets}
        if HAS_IPV4:
            assert socket.AF_INET in families
        if HAS_IPV6:
            assert socket.AF_INET6 in families
        assert {sock.getsockname()[1] for sock in sockets} == {port}
        if hasattr(socket, "IPPROTO_IPV6"):
            for sock in sockets:
                if sock.family == socket.AF_INET6:
                    assert sock.getsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY) == 1
    finally:
        _close_all(sockets)


def test_unspecified_host_shares_ephemeral_port():
    """With port 0, every bound family ends up on the same ephemeral port."""
    sockets = create_server_sockets((None, 0), reuse_port=False)
    try:
        assert len({sock.getsockname()[1] for sock in sockets}) == 1
    finally:
        _close_all(sockets)


@pytest.mark.skipif(not HAS_IPV4, reason="requires IPv4")
def test_unsupported_family_is_skipped(monkeypatch: pytest.MonkeyPatch):
    """An IPv6 entry from the resolver must not crash startup on hosts where
    the IPv6 socket cannot even be created (e.g. IPv6-disabled containers)."""
    real_socket = socket.socket

    def fake_socket(family=socket.AF_INET, *args, **kwargs):
        if family == socket.AF_INET6:
            raise OSError(errno.EAFNOSUPPORT, "Address family not supported")
        return real_socket(family, *args, **kwargs)

    monkeypatch.setattr(socket, "socket", fake_socket)
    port, sockets = _create_sockets_on_open_port(None)
    try:
        assert sockets
        assert all(sock.family == socket.AF_INET for sock in sockets)
    finally:
        _close_all(sockets)


@pytest.mark.skipif(not HAS_IPV4, reason="requires IPv4")
def test_unbindable_family_is_skipped(monkeypatch: pytest.MonkeyPatch):
    """EADDRNOTAVAIL on one family's bind skips it instead of failing."""

    class V6BindFails(socket.socket):
        def bind(self, addr):
            if self.family == socket.AF_INET6:
                raise OSError(errno.EADDRNOTAVAIL, "Cannot assign requested address")
            super().bind(addr)

    monkeypatch.setattr(socket, "socket", V6BindFails)
    port, sockets = _create_sockets_on_open_port(None)
    try:
        assert sockets
        assert all(sock.family == socket.AF_INET for sock in sockets)
    finally:
        _close_all(sockets)


def test_duplicate_addrinfos_deduplicated(monkeypatch: pytest.MonkeyPatch):
    """Resolvers returning duplicate entries must not double-bind."""
    real_getaddrinfo = socket.getaddrinfo

    def fake_getaddrinfo(*args, **kwargs):
        infos = real_getaddrinfo(*args, **kwargs)
        return infos + infos

    monkeypatch.setattr(socket, "getaddrinfo", fake_getaddrinfo)
    port, sockets = _create_sockets_on_open_port(None)
    try:
        # One socket per family; a duplicate bind would have raised.
        families = [sock.family for sock in sockets]
        assert len(families) == len(set(families))
    finally:
        _close_all(sockets)


@pytest.mark.skipif(
    not (HAS_IPV4 and HAS_IPV6), reason="requires both address families"
)
def test_partial_bind_failure_closes_earlier_sockets():
    """If one family's bind fails hard (EADDRINUSE), sockets already bound
    for other families must be closed before the error propagates."""
    infos = list(
        dict.fromkeys(
            socket.getaddrinfo(
                None, 0, type=socket.SOCK_STREAM, flags=socket.AI_PASSIVE
            )
        )
    )
    families = [info[0] for info in infos]
    if len(families) < 2:
        pytest.skip("resolver returned a single family")

    def bind_wildcard(family: socket.AddressFamily, port: int) -> socket.socket:
        sock = socket.socket(family, socket.SOCK_STREAM)
        if family == socket.AF_INET6:
            sock.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 1)
        sock.bind(("", port))
        return sock

    # Occupy the wildcard for whichever family binds second, so the first
    # family binds successfully before the failure.
    for _ in range(5):
        port = get_open_port()
        try:
            blocker = bind_wildcard(families[1], port)
        except OSError:
            continue
        break
    else:
        pytest.fail("could not find a free port")

    try:
        with pytest.raises(OSError):
            create_server_sockets((None, port), reuse_port=False)

        # The first family's socket must have been closed on failure,
        # leaving its wildcard immediately bindable again.
        recheck = bind_wildcard(families[0], port)
        recheck.close()
    finally:
        blocker.close()
