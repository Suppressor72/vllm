# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the adaptive-verification width floor and selector provider.

Covers the concurrency-gated width-floor reservation, the reserve-then-rank
allocation kernel, and the DFlash2 selector acceptance provider — all
CPU-only, no engine boot, no GPU.
"""

import numpy as np
import torch

from vllm.v1.worker.gpu.spec_decode.adaptive_verification import (
    _assign_draft_token_budget,
    _min_width_reservation,
)
from vllm.v1.worker.gpu.spec_decode.dflash2.speculator import (
    _SELECTOR_CAL_KNOTS,
    _SELECTOR_CAL_VALS,
    apply_selector_calibration,
    selector_acceptance_confidences,
)

RNG = np.random.default_rng(20260829)


# ---------------- floor reservation + allocation kernel ----------------


def test_floor_reservation_semantics():
    sched = np.array([7, 7, 7, 0, 7], dtype=np.int32)
    assert not _min_width_reservation(sched, 0, 2, 100).any()
    assert not _min_width_reservation(sched, 7, 2, 100).any()  # 4 verifying > 2
    r = _min_width_reservation(np.array([7, 3, 0], dtype=np.int32), 5, 8, 100)
    assert r.tolist() == [5, 3, 0]
    r = _min_width_reservation(np.array([7, 7], dtype=np.int32), 7, 2, 8)
    assert r.tolist() == [4, 4]  # logits-cap degradation


def _run_kernel(conf, scheduled, budget, reserved=None):
    caps = torch.tensor(scheduled, dtype=torch.int32)
    idx = torch.arange(len(scheduled), dtype=torch.int32)
    r = torch.tensor(reserved, dtype=torch.int32) if reserved is not None else None
    total = int(sum(reserved)) if reserved is not None else 0
    _assign_draft_token_budget(conf, idx, caps, budget, conf.shape[1], r, total)
    return caps


def test_kernel_identity_without_reservation():
    g = torch.Generator().manual_seed(7)
    conf = torch.rand(4, 7, generator=g, dtype=torch.float32) * 0.9 + 0.05
    sched = [7, 5, 3, 7]
    for budget in (1, 6, 13, 22):
        caps = _run_kernel(conf, sched, budget)
        # Reference: the shipped pure top-k.
        survival = conf.cumprod(dim=1)
        steps = torch.arange(7)
        oor = steps[None, :] >= torch.tensor(sched)[:, None]
        survival = survival.masked_fill(oor, -float("inf"))
        winners = survival.flatten().topk(budget).indices
        admitted = torch.zeros(28, dtype=torch.bool).index_fill_(0, winners, True)
        ref = admitted.view(4, 7).sum(1).to(torch.int32)
        assert torch.equal(caps, ref), (budget, caps, ref)


def test_kernel_reservation_honored():
    g = torch.Generator().manual_seed(7)
    conf = torch.rand(4, 7, generator=g, dtype=torch.float32) * 0.9 + 0.05
    conf[0, :] = 0.01  # adversarial low-confidence request
    caps = _run_kernel(conf, [7, 7, 7, 7], 16, [3, 3, 3, 3])
    assert int(caps[0]) >= 3  # keeps its floor
    assert int(caps.sum()) == 16  # totals conserved
    caps = _run_kernel(conf, [7, 7, 7, 7], 12, [3, 3, 3, 3])
    assert caps.tolist() == [3, 3, 3, 3]  # budget == sum(R): exact


def test_budget_geq_reservations():
    sched = np.array([7, 7, 7], dtype=np.int32)
    r = _min_width_reservation(sched, 7, 2, 21)
    for argmax_b in (0, 3, 12, 21):
        budget = max(argmax_b, min(21, int(r.sum())))
        assert budget >= int(r.sum()) and budget <= 21


# ---------------- selector provider + calibration ----------------


def test_selector_confidences_shape_and_range():
    scores = torch.randn(4, 7, 16)
    p1 = selector_acceptance_confidences(scores)
    assert p1.shape == (4, 7)
    assert bool((p1 > 0).all()) and bool((p1 <= 1).all())
    # top-1 of a softmax is always >= 1/K
    assert bool((p1 >= 1.0 / 16 - 1e-6).all())


def test_selector_calibration_monotone_and_clamped():
    knots = torch.tensor(_SELECTOR_CAL_KNOTS, dtype=torch.float32)
    vals = torch.tensor(_SELECTOR_CAL_VALS, dtype=torch.float32)
    assert bool((torch.diff(vals) >= 0).all())  # monotone table
    x = torch.linspace(0.0, 1.0, 1001)
    y = apply_selector_calibration(x, knots, vals)
    assert bool((torch.diff(y) >= 0).all())  # monotone map
    assert y[0] == vals[0] and y[-1] == vals[-1]  # clamped
    # inside the table: piecewise-constant steps at the knots
    assert y[500] <= y[700] <= y[900]


def test_selector_calibration_matches_fitted_endpoints():
    knots = torch.tensor(_SELECTOR_CAL_KNOTS, dtype=torch.float32)
    vals = torch.tensor(_SELECTOR_CAL_VALS, dtype=torch.float32)
    y = apply_selector_calibration(torch.tensor([0.0, 0.3, 0.99, 1.0]), knots, vals)
    assert y[0] == vals[0]
    assert y[3] == vals[-1]
    assert bool((y[1:3] >= vals[0]).all()) and bool((y[1:3] <= vals[-1]).all())
