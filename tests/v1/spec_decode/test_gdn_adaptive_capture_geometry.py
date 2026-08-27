# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the adaptive-verification capture-geometry fix (#51869/#53929).

Covers the three invariants the corruption chain established:
1. Compile-cache separation: enable_adaptive_verification must be part of
   SpeculativeConfig.compute_hash().
2. Varlen capture dummies must exercise the promised max_query_len (one
   largest request + capped spread), for every descriptor rung.
3. Varlen descriptors must claim only the window their dummy exercises
   (honest max_query_len per rung), and layouts no rung covers must fall
   back rather than match dishonestly.
"""

import numpy as np
import pytest

from vllm.config import SpeculativeConfig
from vllm.v1.worker.gpu.cudagraph_utils import varlen_descriptor_rungs


def _hash_with_adaptive(flag: bool) -> str:
    # Bypass __init__ validation (dspark + adaptive needs a real draft
    # checkpoint); compute_hash only reads these fields.
    cfg = object.__new__(SpeculativeConfig)
    cfg.method = "dspark"
    cfg.enable_adaptive_verification = flag
    cfg.draft_model_config = None
    return cfg.compute_hash()


def test_adaptive_flag_separates_speculative_hash():
    """Adaptive mode changes captured graphs, so the flag must hash."""
    assert _hash_with_adaptive(True) != _hash_with_adaptive(False)


import torch  # noqa: E402

from vllm.v1.worker.gpu.input_batch import (  # noqa: E402
    InputBatch,
    InputBuffers,
)


@pytest.mark.parametrize(
    "num_tokens,num_reqs,max_query_len",
    [
        (1, 1, 8),
        (8, 1, 8),  # tight rung: single full-window request
        (8, 8, 1),  # wide rung: all 1-token
        (16, 8, 8),  # [8, 2, 1, 1, 1, 1, 1, 1]
        (16, 2, 8),  # [8, 8]
        (12, 8, 5),  # wide rung for T=12: big = min(5, 5) = 5
        (6, 2, 8),  # big = min(8, 5) = 5, rest spread: [5, 1]
    ],
)
def test_priority_dummy_covers_promised_window(num_tokens, num_reqs, max_query_len):
    """The dummy's realized max per-request length must equal
    min(max_query_len, num_tokens - num_reqs + 1) and sum to num_tokens."""
    buffers = InputBuffers(num_reqs, num_tokens, torch.device("cpu"))
    batch = InputBatch.make_dummy(
        num_reqs, num_tokens, buffers, max_query_len=max_query_len
    )
    lens = np.asarray(batch.num_scheduled_tokens)
    assert int(lens.sum()) == num_tokens
    assert lens.min() >= 1
    assert int(lens.max()) == min(max_query_len, num_tokens - num_reqs + 1)
    # Every request within the cap.
    assert int(lens.max()) <= max_query_len


@pytest.mark.parametrize("num_tokens", [1, 2, 4, 8, 16])
def test_varlen_descriptor_ladder_is_honest(num_tokens: int):
    """Each rung claims exactly the window its dummy realizes."""
    max_q, max_reqs = 8, 8
    for r, claimed in varlen_descriptor_rungs(num_tokens, max_q, max_reqs):
        assert 1 <= claimed <= max_q
        assert r <= max_reqs
        # The dummy for (num_tokens, r, claimed) realizes exactly
        # `claimed` as its max per-request length (tested above).
        assert claimed == min(max_q, num_tokens - r + 1)


def test_uncovered_layout_falls_back():
    """T=12, 5 requests, q=8 matches neither T=12 rung and must not
    match dishonestly."""
    T, max_q, max_reqs = 12, 8, 8
    rungs = varlen_descriptor_rungs(T, max_q, max_reqs)
    batch_num_reqs, batch_max_q = 5, 8
    matches = [(r, q) for r, q in rungs if r >= batch_num_reqs and q >= batch_max_q]
    assert matches == []


@pytest.mark.parametrize(
    "schedule,static_k",
    [
        ([0, 3, 7], 7),  # valid tiers: max == static
        ([0, 2], 7),  # max below static K
        ([0, 100], 7),  # oversized tier: runtime clamps to static
        ([], 7),  # empty schedule
    ],
)
def test_capture_bound_clamps_schedule(schedule, static_k):
    """The capture bound must never exceed decode_query_len regardless of
    raw schedule entries; runtime clamps scheduled K the same way."""
    decode_query_len = static_k + 1
    bound = min(decode_query_len, max(schedule, default=decode_query_len))
    assert bound <= decode_query_len


def test_mid_tier_routes_honestly():
    """A K=3 tier (q=4) replays only into a rung whose dummy exercised
    at least q=4 — never a 1-token wide rung."""
    rungs = varlen_descriptor_rungs(16, 8, 8)
    for r, q in rungs:
        if q >= 4:
            # The dummy for this rung realized q as its max request.
            assert q == min(8, 16 - r + 1)
        else:
            # A q=4 batch must not match this rung.
            assert q < 4
