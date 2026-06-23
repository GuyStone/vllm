# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Equivalence tests for the engine-native beam search core (BeamGroupState).

Drives both BeamGroupState and a NumPy reference that mirrors the client-side
algorithm in vllm/entrypoints/generate/beam_search/online.py from the SAME
deterministic logprob oracle, and asserts the final beams are identical. This locks
the selection logic (expand / EOS routing / top-k prune / length-penalty ranking)
before the engine wiring is built.
"""

from __future__ import annotations

import numpy as np
import pytest

from vllm.v1.engine.beam_search_group import (
    BeamGroupState,
    BeamSearchManager,
    get_beam_search_score,
)

VOCAB = 1000
NEG_INF = float("-inf")


def make_oracle(seed: int, eos_token_id: int | None, eos_prob: float, n_cand: int):
    """Deterministic 'engine': maps a beam's generated tokens -> (token_ids, logprobs).

    Same tokens always yield the same candidates, so two equivalent search
    implementations that hold the same beams see identical inputs each step.
    """

    def oracle(tokens: list[int]) -> tuple[list[int], list[float]]:
        rng = np.random.default_rng(seed + hash(tuple(tokens)) % (2**31))
        # Distinct token ids (avoids dict-key collisions, mirrors logprobs.keys()).
        token_ids = rng.choice(VOCAB, size=n_cand, replace=False).tolist()
        # Distinct logprobs (avoids tie-ordering ambiguity between argpartition/sort).
        logprobs = (
            -rng.permutation(n_cand).astype(float) - rng.random(n_cand)
        ).tolist()
        if eos_token_id is not None and rng.random() < eos_prob:
            # Force an EOS candidate at a random slot.
            token_ids[0] = eos_token_id
        return token_ids, logprobs

    return oracle


def reference_search(
    oracle, beam_width, max_tokens, eos_token_id, ignore_eos, length_penalty
):
    """Faithful re-implementation of online.py's loop (NumPy argpartition path)."""
    # beams: list of (tokens, cum_logprob)
    beams = [([], 0.0)]
    completed: list[tuple[list[int], float]] = []
    for _ in range(max_tokens):
        all_tok, all_lp, parent = [], [], []
        for bi, (toks, cum) in enumerate(beams):
            cand_tok, cand_lp = oracle(toks)
            for tk, lp in zip(cand_tok, cand_lp):
                all_tok.append(tk)
                all_lp.append(cum + lp)
                parent.append(bi)
        all_tok = np.array(all_tok)
        all_lp = np.array(all_lp, dtype=float)
        parent = np.array(parent)

        if not ignore_eos and eos_token_id is not None:
            eos_idx = np.where(all_tok == eos_token_id)[0]
            for idx in eos_idx:
                completed.append(
                    (beams[parent[idx]][0] + [eos_token_id], float(all_lp[idx]))
                )
            all_lp[eos_idx] = NEG_INF

        topn = np.argpartition(np.negative(all_lp), beam_width)[:beam_width]
        beams = [
            (beams[parent[i]][0] + [int(all_tok[i])], float(all_lp[i])) for i in topn
        ]
        if not beams:
            break

    def score(item):
        toks, cum = item
        seq_len = len(toks)
        if seq_len and eos_token_id is not None and toks[-1] == eos_token_id:
            seq_len -= 1
        return get_beam_search_score(max(seq_len, 1), cum, length_penalty)

    pool = completed + beams
    pool.sort(key=score, reverse=True)
    return pool[:beam_width]


def run_beam_group(
    oracle, beam_width, max_tokens, eos_token_id, ignore_eos, length_penalty
):
    state = BeamGroupState(
        beam_width=beam_width,
        max_tokens=max_tokens,
        eos_token_id=eos_token_id,
        length_penalty=length_penalty,
        ignore_eos=ignore_eos,
    )
    done = False
    while not done:
        per_tok, per_lp = [], []
        for beam in state.beams:
            t, lp = oracle(beam.tokens)
            per_tok.append(t)
            per_lp.append(lp)
        done = state.step(per_tok, per_lp)
    return [(b.tokens, b.cum_logprob) for b in state.finalize()]


@pytest.mark.parametrize("beam_width", [2, 4, 8])
@pytest.mark.parametrize("max_tokens", [1, 4, 8])
@pytest.mark.parametrize("ignore_eos", [True, False])
@pytest.mark.parametrize("length_penalty", [1.0, 0.0, 2.0])
@pytest.mark.parametrize("seed", [0, 1, 7])
def test_beam_group_matches_reference(
    beam_width, max_tokens, ignore_eos, length_penalty, seed
):
    eos = 42
    oracle = make_oracle(seed, eos, eos_prob=0.3, n_cand=2 * beam_width)

    ref = reference_search(
        oracle, beam_width, max_tokens, eos, ignore_eos, length_penalty
    )
    got = run_beam_group(
        oracle, beam_width, max_tokens, eos, ignore_eos, length_penalty
    )

    # Final beams must match in both token sequence and (close) score.
    assert [t for t, _ in got] == [t for t, _ in ref], (
        f"seqs differ: got={[t for t, _ in got]} ref={[t for t, _ in ref]}"
    )
    for (gt, gs), (rt, rs) in zip(got, ref):
        assert gt == rt
        assert gs == pytest.approx(rs, abs=1e-9)


class _FakeLogprobs:
    """Duck-typed LogprobsLists: row-0 holds this step's candidate (ids, logprobs)."""

    def __init__(self, token_ids, logprobs):
        self.logprob_token_ids = np.asarray([token_ids])
        self.logprobs = np.asarray([logprobs], dtype=float)


class _FakeBase:
    prompt_token_ids: list[int] = []


def _direct_pool(oracle, beam_width, max_tokens, eos, ignore_eos, length_penalty):
    """Run BeamGroupState directly; return the full (completed + live) pool."""
    state = BeamGroupState(
        beam_width=beam_width, max_tokens=max_tokens, eos_token_id=eos,
        length_penalty=length_penalty, ignore_eos=ignore_eos,
    )
    done = False
    while not done:
        per_tok, per_lp = [], []
        for beam in state.beams:
            t, lp = oracle(beam.tokens)
            per_tok.append(t)
            per_lp.append(lp)
        done = state.step(per_tok, per_lp)
    return [(b.tokens, b.cum_logprob) for b in (state.completed + state.beams)]


def _run_manager(oracles, beam_width, max_tokens, eos, ignore_eos, length_penalty):
    """Drive BeamSearchManager event-style (as the engine would) for all groups
    interleaved; return {group_id: pool}. make_child returns (req_id, tokens)."""
    mgr = BeamSearchManager(make_child=lambda base, rid, toks: (rid, toks))
    current = []  # list of (req_id, tokens) currently in flight
    for gid, _ in oracles.items():
        current += mgr.add_group(
            gid, _FakeBase(), beam_width, max_tokens, length_penalty, ignore_eos, eos
        )
    results: dict = {}
    while mgr.active:
        # Empty-prompt fakes: tokens == generated tokens, so oracle(tokens) works.
        for req_id, toks in current:
            gid = req_id.split("::")[0]
            tok, lp = oracles[gid](toks)
            mgr.on_output(req_id, _FakeLogprobs(tok, lp), finished=True)
        _abort, current = mgr.advance()
        results.update(mgr.pop_finished())
    return results


@pytest.mark.parametrize("beam_width", [2, 4, 8])
@pytest.mark.parametrize("max_tokens", [1, 4, 8])
@pytest.mark.parametrize("ignore_eos", [True, False])
@pytest.mark.parametrize("seed", [0, 3])
def test_manager_matches_direct_loop(beam_width, max_tokens, ignore_eos, seed):
    eos, lp = 42, 1.0
    oracle = make_oracle(seed, eos, eos_prob=0.3, n_cand=2 * beam_width)
    direct = _direct_pool(oracle, beam_width, max_tokens, eos, ignore_eos, lp)
    got = _run_manager({"g0": oracle}, beam_width, max_tokens, eos, ignore_eos, lp)
    assert got["g0"] == direct


def test_manager_multi_group_interleaved():
    """Several groups driven concurrently must each match their solo direct loop."""
    bw, mt, eos, lp = 4, 6, 42, 1.0
    oracles = {
        f"g{i}": make_oracle(i, eos, eos_prob=0.3, n_cand=2 * bw) for i in range(4)
    }
    got = _run_manager(oracles, bw, mt, eos, False, lp)
    assert set(got) == set(oracles)
    for gid, oracle in oracles.items():
        assert got[gid] == _direct_pool(oracle, bw, mt, eos, False, lp)


def test_parent_idx_tracked_for_reparenting():
    """M2 KV re-parenting consumes parent_idx; ensure it points into the prior beams."""
    state = BeamGroupState(beam_width=3, max_tokens=3, eos_token_id=None)
    # 1 live beam -> all survivors must have parent_idx 0 after step 1.
    state.step([[1, 2, 3, 4, 5, 6]], [[-0.1, -0.2, -0.3, -0.4, -0.5, -0.6]])
    assert len(state.beams) == 3
    assert all(b.parent_idx == 0 for b in state.beams)
    # Step 2: 3 live beams -> parent_idx must index 0..2.
    per_tok = [[10, 11, 12, 13, 14, 15] for _ in range(3)]
    per_lp = [[-0.1, -0.2, -0.3, -0.4, -0.5, -0.6] for _ in range(3)]
    state.step(per_tok, per_lp)
    assert all(0 <= b.parent_idx < 3 for b in state.beams)
