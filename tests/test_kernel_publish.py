"""Unit tests for the consolidated bus-publish path in FunSearchKernel.

The per-kernel ``_publish`` / ``_find_nervous_bin`` copies were merged into the
base class; these tests exercise that single path with a mocked ``nervous``
binary so the publish layer is covered WITHOUT a live run. This closes the gap
that let a `_git_commit_short` import error ship undetected (it only fired in
excluded live tests).

Marked not-live: no network, no sandbox, no real nervous CLI.
"""

from __future__ import annotations

import json
import os
from types import SimpleNamespace
from unittest import mock

from autobench.kernels import KernelConfig
from autobench.kernels.base import FunSearchKernel


class _StubKernel(FunSearchKernel):
    """Minimal concrete kernel — implements the abstract surface only."""

    BUS_CHANNEL_PREFIX = "stub"

    def load_instances(self):
        return []

    def evaluate_candidate(self, code, instance):
        return 0.0

    def build_prompt(self, island, top_programs, generation, hint=""):
        return ""

    def parse_response(self, response):
        return ""

    def seed_programs(self, island_id, generation):
        return []


def _make(nervous_bin=None):
    """Build a stub kernel without the heavy __init__ (no sandbox/prior)."""
    k = _StubKernel.__new__(_StubKernel)
    k.config = KernelConfig(
        instances=["demo"], n_islands=2, population_per_island=3,
        generations=5, plateau_generations=5, temperature=0.9,
        plateau_hint=True, bus_verbose=False,
    )
    k.run_id = "01TESTRUN"
    k.generation = 4
    k.stop_reason = "test stop"
    k.llm_requests = 7
    k._nervous_bin = nervous_bin
    return k


def _events(home):
    p = home / ".cache" / "nervous-bus" / "debug.jsonl"
    return [json.loads(ln) for ln in p.read_text().splitlines() if ln.strip()]


def test_publish_writes_cloudevents_envelope(tmp_path):
    k = _make()
    with mock.patch("autobench.kernels.base.Path.home", return_value=tmp_path):
        assert k._publish("stub.kernel.started.v1", {"run_id": "01TESTRUN", "x": 1}) is True
    evs = _events(tmp_path)
    assert len(evs) == 1
    e = evs[0]
    assert e["specversion"] == "1.0"
    assert e["type"] == "stub.kernel.started.v1"
    assert e["source"] == "/autobench/stub_kernel"   # derived from BUS_CHANNEL_PREFIX
    assert e["datacontenttype"] == "application/json"
    assert e["data"] == {"run_id": "01TESTRUN", "x": 1}


def test_publish_invokes_nervous_cli_when_bin_present(tmp_path):
    k = _make(nervous_bin="/fake/nervous")
    with mock.patch("autobench.kernels.base.Path.home", return_value=tmp_path), \
         mock.patch("autobench.kernels.base.subprocess.Popen") as popen:
        popen.return_value.communicate.return_value = (b"", b"")
        k._publish("stub.generation.completed.v1", {"g": 1})
    assert popen.called
    # --json pass-through: the already-built envelope is fed on stdin verbatim,
    # NOT passed as an argv payload (which the non-json path would re-wrap).
    argv = popen.call_args[0][0]
    assert argv == ["/fake/nervous", "publish", "--json"]
    stdin_payload = popen.return_value.communicate.call_args[0][0]
    sent = json.loads(stdin_payload)
    assert sent["type"] == "stub.generation.completed.v1"
    assert sent["source"] == "/autobench/stub_kernel"  # not re-sourced from CWD
    assert sent["data"] == {"g": 1}
    # Zellij fan-out off; debug.jsonl write redirected so nervous can't double-write;
    # Redis left ENABLED (NO_REDIS not forced) — this is the only Redis path.
    env = popen.call_args.kwargs["env"]
    assert env["NERVOUS_NO_ZELLIJ"] == "1"
    assert env["NERVOUS_DEBUG_LOG"] == os.devnull
    assert env.get("NERVOUS_NO_REDIS") != "1"


def test_publish_skips_cli_when_no_bin(tmp_path):
    k = _make(nervous_bin=None)
    with mock.patch("autobench.kernels.base.Path.home", return_value=tmp_path), \
         mock.patch("autobench.kernels.base.subprocess.Popen") as popen:
        k._publish("stub.island_reset.v1", {})
    assert not popen.called  # durable debug.jsonl write still happened
    assert _events(tmp_path)[0]["type"] == "stub.island_reset.v1"


def test_publish_started_default_payload(tmp_path):
    k = _make()
    with mock.patch("autobench.kernels.base.Path.home", return_value=tmp_path), \
         mock.patch("autobench.kernels.base._git_commit_short", return_value="abc1234"):
        k._publish_started()
    d = _events(tmp_path)[0]
    assert d["type"] == "stub.kernel.started.v1"
    assert d["data"]["git_commit"] == "abc1234"
    assert d["data"]["instances"] == ["demo"]
    assert d["data"]["n_islands"] == 2
    assert d["data"]["generations"] == 5


def test_publish_completed_default_payload(tmp_path):
    k = _make()
    best = SimpleNamespace(id="p1", fitness=0.83, island=1, generation=3, source="llm")
    with mock.patch("autobench.kernels.base.Path.home", return_value=tmp_path):
        k._publish_completed([best])
    d = _events(tmp_path)[0]
    assert d["type"] == "stub.kernel.completed.v1"
    assert d["data"]["total_generations"] == 4
    assert d["data"]["stop_reason"] == "test stop"
    assert d["data"]["llm_requests"] == 7
    assert d["data"]["best_program"]["fitness"] == 0.83
    assert d["data"]["best_program"]["island"] == 1


def test_publish_completed_handles_empty(tmp_path):
    k = _make()
    with mock.patch("autobench.kernels.base.Path.home", return_value=tmp_path):
        k._publish_completed([])
    assert _events(tmp_path)[0]["data"]["best_program"] is None


def test_all_registered_kernels_have_a_concrete_prefix():
    """Regression guard: every kernel must set BUS_CHANNEL_PREFIX (not the
    base default 'kernel'), else it emits malformed kernel.* channels."""
    import autobench.kernels.cli  # noqa: F401 — registers all kernels
    from autobench.kernels import KERNEL_REGISTRY
    offenders = {
        name: cls.BUS_CHANNEL_PREFIX
        for name, cls in KERNEL_REGISTRY.items()
        if cls.BUS_CHANNEL_PREFIX in ("kernel", "")
    }
    assert not offenders, f"kernels missing a real BUS_CHANNEL_PREFIX: {offenders}"
