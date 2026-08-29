# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the adaptive-verification history estimator and floor.

Covers the censoring-aware conditional estimator (AcceptanceHistoryEstimator),
the concurrency-gated width-floor reservation, and the reserve-then-rank
allocation kernel — all CPU-only, no engine boot, no GPU.

Estimator ground truth: accepted lengths are drawn from a KNOWN conditional
ladder q* through randomized trim (admitted-width) schedules. The primary
ladder is strictly positive and decaying (a zero-tail min-variance ladder
cannot exercise tail-censoring recovery); the min-variance ladder runs as a
second case.
"""

import numpy as np
import pytest
import torch

from vllm.v1.worker.gpu.spec_decode.adaptive_verification import (
    AcceptanceHistoryEstimator,
    _assign_draft_token_budget,
    _min_width_reservation,
)

RNG = np.random.default_rng(20260829)


def draw_accepted(q_star: np.ndarray) -> int:
    """Draw an accepted prefix length from conditional rates q*."""
    accepted = 0
    for q in q_star:
        if RNG.random() < q:
            accepted += 1
        else:
            break
    return accepted


def run_and_average(
    q_star: np.ndarray, steps: int, trim_schedule, alpha: float
) -> np.ndarray:
    """Snapshot-average of the conditionals over the trajectory's second
    half: a stationary constant-alpha EMA keeps fluctuating (std ~
    sqrt(alpha/(2-alpha) * q(1-q))), so unbiasedness is tested through
    the time-average, which converges like a mean."""
    est = AcceptanceHistoryEstimator(
        num_slots=1,
        num_steps=len(q_star),
        alpha=alpha,
        min_count=1,
        warmup_steps=1,
    )
    snaps = []
    for step in range(steps):
        admitted = int(trim_schedule(step))
        accepted = min(draw_accepted(q_star), admitted)
        est.update(0, admitted, accepted)
        if step >= steps // 2 and step % 25 == 0:
            snaps.append(est.conditionals(np.array([0]))[0].copy())
    return np.mean(snaps, axis=0)


@pytest.mark.parametrize(
    "q_star",
    [
        # Primary: strictly positive decaying ladder.
        np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.45, 0.4]),
        # Second case: min-variance zero-tail ladder (AR 2.04 class).
        np.array([1.0, 0.04, 0.0, 0.0, 0.0, 0.0, 0.0]),
    ],
)
def test_ladder_recovery_full_width(q_star):
    est = AcceptanceHistoryEstimator(
        num_slots=1,
        num_steps=len(q_star),
        alpha=0.02,
        min_count=1,
        warmup_steps=1,
    )
    snaps = []
    for step in range(6000):
        est.update(0, len(q_star), min(draw_accepted(q_star), len(q_star)))
        if step >= 3000 and step % 25 == 0:
            snaps.append(est.conditionals(np.array([0]))[0].copy())
    got = np.mean(snaps, axis=0)
    # A position upstream of a hard zero in q* is NEVER reached: it is
    # censored (seeded optimistically at 1), not estimated. Compare
    # only positions the ground truth lets us observe.
    reachable = np.cumprod(np.concatenate(([1.0], q_star[:-1]))) > 0
    # (atol 0.045: residual snapshot-average noise; the forbidden
    # zeros-for-all-p<=w estimator biases the tail by 0.2-0.4.)
    assert np.allclose(got[reachable], q_star[reachable], atol=0.045), (
        got,
        q_star,
    )
    assert np.all(got[~reachable] == 1.0)


def test_censoring_does_not_bias_tail_down():
    """Randomized trims must not bias the recovered ladder downward.

    The forbidden naive estimator (EMA zeros for all p <= admitted) biases
    the tail toward zero; the KM predicate must not.
    """
    q_star = np.array([0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6])
    trims = RNG.integers(1, len(q_star) + 1, size=8000)
    got = run_and_average(
        q_star, steps=len(trims), trim_schedule=lambda s: trims[s], alpha=0.02
    )
    assert np.allclose(got, q_star, atol=0.04), (got, q_star)


def test_trim_boundary_death_is_censored_not_rejected():
    """accepted == admitted must never update the next position to zero."""
    est = AcceptanceHistoryEstimator(1, 7, alpha=0.5, min_count=1, warmup_steps=1)
    # Repeated boundary deaths at width 3 with everything accepted.
    for _ in range(50):
        est.update(0, admitted=3, accepted=3)
    # Positions 1..3 updated (reached and verified); position 4 censored.
    assert est._pool_den[3] == 0  # p=4 never observed
    # A later full-width full-accept step observes position 4 as a SURVIVOR.
    est.update(0, admitted=7, accepted=7)
    assert est._pool_num[3] == 1 and est._pool_den[3] == 1


def test_no_absorbing_zero_width():
    """admitted=0 is a no-op, not a death; recovery resumes after."""
    est = AcceptanceHistoryEstimator(1, 7, alpha=0.05, min_count=1, warmup_steps=2)
    for _ in range(10):
        est.update(0, admitted=0, accepted=0)
    assert est._steps[0] == 0 and est.is_cold(np.array([0]))[0]
    q_star = np.array([0.8, 0.7, 0.6, 0.5, 0.5, 0.5, 0.5])
    snaps = []
    for i in range(4000):
        est.update(0, 7, min(draw_accepted(q_star), 7))
        if i >= 2000 and i % 25 == 0:
            snaps.append(est.conditionals(np.array([0]))[0].copy())
    assert np.allclose(np.mean(snaps, axis=0), q_star, atol=0.05)


def test_warmup_and_seed():
    """Cold requests shrink toward the pooled seed; empty pool seeds q=1."""
    est = AcceptanceHistoryEstimator(2, 4, alpha=0.2, min_count=8, warmup_steps=5)
    # Slot 0 accumulates evidence; slot 1 stays cold.
    q_star = np.array([0.9, 0.7, 0.5, 0.3])
    for _ in range(200):
        est.update(0, 4, min(draw_accepted(q_star), 4))
    assert not est.is_cold(np.array([0]))[0]
    assert est.is_cold(np.array([1]))[0]
    cold_q = est.conditionals(np.array([1]))[0]
    # Empty-history slot sees only the pooled seed (all ~q_star-ish, in
    # (0, 1], never zeros).
    assert np.all(cold_q > 0.0) and np.all(cold_q <= 1.0)
    # A fully empty pool seeds full-width bias (q = 1).
    fresh = AcceptanceHistoryEstimator(1, 4, alpha=0.2, min_count=8, warmup_steps=5)
    assert np.all(fresh.conditionals(np.array([0]))[0] == 1.0)


def test_reset_clears_slot():
    est = AcceptanceHistoryEstimator(1, 4, alpha=0.3, min_count=1, warmup_steps=1)
    est.update(0, 4, 0)
    est.reset(0)
    assert est._steps[0] == 0 and np.all(est._ema[0] == 1.0)


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
