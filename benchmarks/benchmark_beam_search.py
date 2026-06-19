# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark offline beam-search latency across beam widths.

This was used to diagnose and validate the beam-search logprob
detokenization bottleneck: beam search requests ``2 * beam_width`` logprobs
per step, and the engine used to detokenize every one of those token ids
into a string even though beam search only consumes the logprob floats. That
work was O(beam_width^2) per step and dominated wall-clock at wide beam
widths (the regime used for recommendation candidate generation).

The fix (``detokenize=False`` on the internal beam-search ``SamplingParams``)
removes the decode loop. For the online / under-load picture, use the
``beam_search`` profiles in Spotify's ``lpm-benchmark`` instead; this script
measures the single-request offline path.

Example:
    python benchmarks/benchmark_beam_search.py \
        --model Qwen/Qwen3-1.7B --beam-widths 15,30,100,250 \
        --prompt-len 2000 --output-tokens 5
"""

import argparse
import statistics
import time

from vllm import LLM
from vllm.sampling_params import BeamSearchParams


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
    parser.add_argument(
        "--beam-widths",
        default="15,30,100,250",
        help="Comma-separated beam widths to sweep.",
    )
    parser.add_argument("--prompt-len", type=int, default=2000)
    parser.add_argument("--output-tokens", type=int, default=5)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--enforce-eager", action="store_true")
    args = parser.parse_args()

    beam_widths = [int(x) for x in args.beam_widths.split(",")]
    max_beam_width = max(beam_widths)

    # Deterministic token prompt of an exact length (avoids tokenizer variance
    # across runs and keeps the shared prefix identical across beams).
    prompt = {"prompt_token_ids": [100 + (i % 1000) for i in range(args.prompt_len)]}

    llm = LLM(
        model=args.model,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=args.enforce_eager,
        # Beam search internally requests 2 * beam_width logprobs.
        max_logprobs=2 * max_beam_width,
    )

    print(
        f"# model={args.model} prompt_len={args.prompt_len} "
        f"output_tokens={args.output_tokens} iters={args.iters}"
    )
    for beam_width in beam_widths:
        params = BeamSearchParams(beam_width=beam_width, max_tokens=args.output_tokens)
        llm.beam_search([prompt], params)  # warmup
        timings = []
        for _ in range(args.iters):
            start = time.perf_counter()
            llm.beam_search([prompt], params)
            timings.append(time.perf_counter() - start)
        print(
            f"beam_width={beam_width:4d}  "
            f"median={statistics.median(timings):7.4f}s  "
            f"min={min(timings):7.4f}s"
        )


if __name__ == "__main__":
    main()
