# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Engine-native beam search: the per-step expand/prune core.

This module holds the *algorithmic heart* of engine-native batched beam search,
deliberately kept free of any engine/scheduler/KV dependencies so it can be unit
tested in isolation and proven equivalent to the existing client-side loop in
``vllm/entrypoints/generate/beam_search/online.py`` before it is wired into the
engine.

Why engine-native (motivation)
------------------------------
Today beam search is a CLIENT-SIDE loop: per output step it submits ``beam_width``
independent requests (``logprobs=2*beam_width``) and expands/prunes in Python. After
the ``detokenize=False`` fix the remaining cost is the per-step round-trips
(client -> AsyncLLM -> ZMQ -> EngineCore -> back): nsys shows the GPU only ~33% busy
because it sits idle in the gap between steps while the client decides the next
beams. Moving the expand/prune *into the engine* keeps the GPU fed.

Rollout (this module is the shared core for both milestones)
------------------------------------------------------------
* **M1 (reuse prefix cache, no KV fork):** run this expand/prune inside the engine
  step loop; feed survivors back by continuing each survivor's token sequence.
  Shared prompt + common generated prefix are reused via prefix caching; only the
  short divergent tail recomputes. Validates the round-trip-elimination win with no
  new KV primitive.
* **M2 (persistent beams + partial-block-copy re-parenting):** keep beams resident
  and re-parent survivors by copying the (sub-block) divergent KV instead of
  recomputing it. ``BeamCandidate.parent_idx`` is the survivor->parent mapping that
  re-parenting consumes; it is populated here so M2 needs no algorithm change.

Equivalence contract
--------------------
The selection logic mirrors ``online.py`` exactly: gather ``2*beam_width`` candidates
per live beam, score by ``cum_logprob + token_logprob``, route EOS candidates to a
completed pool (masking them out of the live top-k), keep the top ``beam_width`` live
survivors, and finally rank completed+live by the HF length-penalty score
(``get_beam_search_score``). This is asserted against a NumPy reference in
``tests/v1/engine/test_beam_search_group.py``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


def get_beam_search_score(
    seq_len: int,
    cumulative_logprob: float,
    length_penalty: float = 1.0,
) -> float:
    """HF-compatible length-penalised score.

    Mirrors ``vllm/entrypoints/generate/beam_search/utils.py``: the EOS token (when
    present) is excluded from ``seq_len`` by the caller before calling this.
    """
    return cumulative_logprob / (seq_len**length_penalty)


@dataclass
class Beam:
    """A live beam: its generated token ids and cumulative logprob.

    ``tokens`` holds only the *generated* tokens (the shared prompt is implicit and
    lives in the engine's request state / KV cache), so length-penalty scoring and
    re-parenting reason about generated length directly.
    """

    tokens: list[int]
    cum_logprob: float
    # Index of this beam within the *previous* step's live-beam list (the parent it
    # was expanded from). -1 for the initial beam. Consumed by M2 KV re-parenting.
    parent_idx: int = -1
    finished: bool = False
    finish_reason: str | None = None


@dataclass
class BeamCandidate:
    """A single (parent_beam, token) expansion considered during a step."""

    parent_idx: int
    token_id: int
    cum_logprob: float


@dataclass
class BeamGroupState:
    """Per-query beam-search state stepped by the engine.

    One instance per beam-search request. ``beams`` are the live beams (initially a
    single empty beam); ``completed`` accumulates finished beams (EOS / max_tokens).
    """

    beam_width: int
    max_tokens: int
    eos_token_id: int | None
    length_penalty: float = 1.0
    ignore_eos: bool = False
    beams: list[Beam] = field(
        default_factory=lambda: [Beam(tokens=[], cum_logprob=0.0)]
    )
    completed: list[Beam] = field(default_factory=list)
    # Number of completed output steps so far.
    step_idx: int = 0

    @property
    def num_live_beams(self) -> int:
        return len(self.beams)

    def _score(self, beam: Beam) -> float:
        seq_len = len(beam.tokens)
        if seq_len:
            last_is_eos = (
                self.eos_token_id is not None
                and beam.tokens[-1] == self.eos_token_id
            )
            if last_is_eos:
                seq_len -= 1
        seq_len = max(seq_len, 1)
        return get_beam_search_score(seq_len, beam.cum_logprob, self.length_penalty)

    def step(
        self,
        per_beam_token_ids: list[list[int]],
        per_beam_logprobs: list[list[float]],
    ) -> bool:
        """Advance one output step.

        Args:
            per_beam_token_ids: for each live beam, its top-``2*beam_width`` candidate
                token ids (the keys of the engine's logprobs output for that beam).
            per_beam_logprobs: matching per-token logprobs (floats).

        Returns:
            ``True`` if the search is finished (no live beams remain or max_tokens
            reached), else ``False``.
        """
        assert len(per_beam_token_ids) == len(self.beams), (
            f"expected candidates for {len(self.beams)} live beams, "
            f"got {len(per_beam_token_ids)}"
        )

        # 1) Expand: every (beam, candidate-token) pair, scored by accumulated logprob.
        candidates: list[BeamCandidate] = []
        for beam_idx, beam in enumerate(self.beams):
            for token_id, logprob in zip(
                per_beam_token_ids[beam_idx], per_beam_logprobs[beam_idx]
            ):
                candidates.append(
                    BeamCandidate(
                        parent_idx=beam_idx,
                        token_id=token_id,
                        cum_logprob=beam.cum_logprob + logprob,
                    )
                )

        # 2) Route EOS candidates to the completed pool (and exclude from the live
        #    top-k), matching online.py's eos masking.
        live_candidates: list[BeamCandidate] = []
        for cand in candidates:
            parent = self.beams[cand.parent_idx]
            if (
                not self.ignore_eos
                and self.eos_token_id is not None
                and cand.token_id == self.eos_token_id
            ):
                self.completed.append(
                    Beam(
                        tokens=parent.tokens + [cand.token_id],
                        cum_logprob=cand.cum_logprob,
                        parent_idx=cand.parent_idx,
                        finished=True,
                        finish_reason="stop",
                    )
                )
            else:
                live_candidates.append(cand)

        # 3) Keep the top ``beam_width`` live candidates by cumulative logprob.
        live_candidates.sort(key=lambda c: c.cum_logprob, reverse=True)
        survivors = live_candidates[: self.beam_width]

        self.beams = [
            Beam(
                tokens=self.beams[c.parent_idx].tokens + [c.token_id],
                cum_logprob=c.cum_logprob,
                parent_idx=c.parent_idx,
            )
            for c in survivors
        ]
        self.step_idx += 1

        # 4) Stop when we've generated max_tokens or no live beams remain.
        return self.step_idx >= self.max_tokens or not self.beams

    def finalize(self) -> list[Beam]:
        """Return the best ``beam_width`` beams (completed + remaining live).

        Ranked by the HF length-penalty score, matching online.py's final sort.
        """
        pool = list(self.completed) + list(self.beams)
        pool.sort(key=self._score, reverse=True)
        return pool[: self.beam_width]


def has_converged_to_finite(scores: list[float]) -> bool:
    """Small helper: True if every score is finite (no all -inf degenerate state)."""
    return all(math.isfinite(s) for s in scores)
