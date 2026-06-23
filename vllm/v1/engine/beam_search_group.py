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

import numpy as np


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

        # 1) Vectorised expand: flatten every (beam, candidate-token) pair into
        #    numpy arrays (avoids materialising O(beam_width * 2*beam_width) Python
        #    objects + a Python sort each step).
        num_beams = len(self.beams)
        lengths = [len(toks) for toks in per_beam_token_ids]
        token_ids = np.concatenate(
            [np.asarray(t, dtype=np.int64) for t in per_beam_token_ids]
        )
        flat_logprobs = np.concatenate(
            [np.asarray(lp, dtype=np.float64) for lp in per_beam_logprobs]
        )
        parent_idx = np.repeat(np.arange(num_beams), lengths)
        base_cum = np.array([b.cum_logprob for b in self.beams], dtype=np.float64)
        cum = np.repeat(base_cum, lengths) + flat_logprobs

        # 2) Route EOS candidates to the completed pool, then mask them to -inf so
        #    they drop out of the live top-k (matches online.py's eos masking).
        if not self.ignore_eos and self.eos_token_id is not None:
            eos_positions = np.nonzero(token_ids == self.eos_token_id)[0]
            for pos in eos_positions:
                parent = self.beams[int(parent_idx[pos])]
                self.completed.append(
                    Beam(
                        tokens=parent.tokens + [self.eos_token_id],
                        cum_logprob=float(cum[pos]),
                        parent_idx=int(parent_idx[pos]),
                        finished=True,
                        finish_reason="stop",
                    )
                )
            cum[eos_positions] = float("-inf")

        # 3) Keep the top ``beam_width`` live candidates by cumulative logprob
        #    (O(n) argpartition + a small sort of just the survivors for order).
        if cum.size <= self.beam_width:
            top = np.argsort(-cum)
        else:
            top = np.argpartition(-cum, self.beam_width)[: self.beam_width]
            top = top[np.argsort(-cum[top])]

        self.beams = [
            Beam(
                tokens=self.beams[int(parent_idx[i])].tokens + [int(token_ids[i])],
                cum_logprob=float(cum[i]),
                parent_idx=int(parent_idx[i]),
            )
            for i in top
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
