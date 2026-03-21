"""Diagnostic: prove Opus + thinking + batch parsing works.

Makes exactly 1 API call using the exact same code path as run_vr().
Shows raw response structure, parsed fields, and extracted data.
"""

from __future__ import annotations

import numpy as np

from synthoracle.agents.vr import (
    _VRBatchResponseSchema,
    _build_iteration_prompt,
    _build_system_prompt,
    _compute_correlations,
    _compute_local_gradients,
    _determine_phase,
)
from synthoracle.optim_utils import compute_reference_point, parse_directions
from synthoracle.oracles.medium import MediumOracle


def main() -> None:
    oracle = MediumOracle()
    thresholds = {"Y3": 0.4}
    batch_size = 3
    seed = 42
    n_initial = 12
    n_iterations = 60

    # Setup exactly as run_vr does
    obj_indices, constraint_indices, signs = parse_directions(oracle)
    reference_point = compute_reference_point(oracle, obj_indices, signs, seed)

    rng = np.random.default_rng(seed)
    lo, hi = oracle.bounds[:, 0], oracle.bounds[:, 1]
    X_all = rng.uniform(lo, hi, size=(n_initial, oracle.n_inputs))
    Y_all = oracle.evaluate_batch(X_all)

    # Build prompt exactly as run_vr does for call_idx=0
    phase = _determine_phase(n_initial, n_initial + n_iterations)
    correlations = _compute_correlations(X_all, Y_all)
    gradients = [
        _compute_local_gradients(oracle, X_all[-k - 1])
        for k in range(min(batch_size, len(X_all)))
    ]
    system_prompt = _build_system_prompt(oracle, thresholds, batch_size)
    prompt = _build_iteration_prompt(
        oracle, X_all, Y_all, [], None, None, 0,
        correlations, gradients, phase, n_initial,
        n_initial + n_iterations, batch_size,
    )

    print("=" * 70)
    print("DIAGNOSTIC: Opus + thinking + batch structured output")
    print("=" * 70)
    print(f"\nSystem prompt length: {len(system_prompt)} chars")
    print(f"User prompt length: {len(prompt)} chars")
    print(f"Phase: {phase}")

    # Make the API call — exactly as run_vr does
    import anthropic
    client = anthropic.Anthropic()

    print("\n--- Making API call ---")
    print(f"Model: claude-opus-4-6")
    print(f"Thinking: adaptive")
    print(f"Output format: _VRBatchResponseSchema")

    msg = client.messages.parse(
        model="claude-opus-4-6",
        max_tokens=16000,
        system=system_prompt,
        messages=[{"role": "user", "content": prompt}],
        temperature=1.0,
        output_format=_VRBatchResponseSchema,
        thinking={"type": "adaptive"},
    )

    # Show raw response structure
    print(f"\n--- Raw response ---")
    print(f"Content blocks: {len(msg.content)}")
    for i, block in enumerate(msg.content):
        block_type = getattr(block, "type", "unknown")
        print(f"  Block {i}: type={block_type}", end="")
        if block_type == "thinking":
            thinking_text = getattr(block, "thinking", "")
            print(f", thinking_length={len(thinking_text)} chars")
            print(f"    Preview: {thinking_text[:200]}...")
        elif block_type == "text":
            text = getattr(block, "text", "")
            print(f", text_length={len(text)} chars")
            print(f"    Preview: {text[:200]}...")
        else:
            print()

    print(f"\nUsage: {msg.usage.input_tokens} input, {msg.usage.output_tokens} output tokens")

    # Show text extraction (the fixed code path)
    raw_text = ""
    for block in msg.content:
        if hasattr(block, "text") and getattr(block, "type", "") == "text":
            raw_text = block.text
            break
    print(f"\nExtracted text block: {len(raw_text)} chars")

    # Show parsed output
    parsed = msg.parsed_output
    print(f"\n--- Parsed output ---")
    print(f"parsed_output is None: {parsed is None}")

    if parsed is not None:
        print(f"reconciliation: {parsed.reconciliation[:100]}...")
        print(f"biggest_surprise: {parsed.biggest_surprise[:100]}...")
        print(f"hypothesis: {parsed.hypothesis[:150]}...")
        print(f"Number of points: {len(parsed.points)}")
        for i, pt in enumerate(parsed.points):
            print(f"\n  Point {i + 1}:")
            print(f"    next_point: {pt.next_point}")
            print(f"    prediction: {pt.prediction}")
            print(f"    explore_or_exploit: {pt.explore_or_exploit}")
            print(f"    reasoning: {pt.reasoning[:100]}...")
            print(f"    falsification: {pt.falsification[:100]}...")

            # Validate
            x = np.array(pt.next_point)
            print(f"    within bounds: {np.all(x >= lo) and np.all(x <= hi)}")
            print(f"    finite: {np.all(np.isfinite(x))}")
            print(f"    prediction finite: {np.all(np.isfinite(pt.prediction))}")

    # Estimate cost
    cost = msg.usage.input_tokens * 15 / 1e6 + msg.usage.output_tokens * 75 / 1e6
    print(f"\n--- Cost ---")
    print(f"This call: ${cost:.4f}")
    print(f"Projected 20-call run: ${cost * 20:.2f}")

    print("\n✓ DIAGNOSTIC PASSED" if parsed is not None and len(parsed.points) == batch_size
          else "\n✗ DIAGNOSTIC FAILED")


if __name__ == "__main__":
    main()
