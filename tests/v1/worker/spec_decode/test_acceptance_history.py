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
from vllm.v1.worker.gpu.spec_decode.dflash2.speculator import (
    _SELECTOR_CAL_KNOTS,
    _SELECTOR_CAL_VALS,
    apply_selector_calibration,
    selector_acceptance_confidences,
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


# ------- Gate-1 defect regressions (each was a real bug) -------


def test_seed_is_not_double_converted():
    """The pooled seed must BE the conditional ladder, not a re-conversion.

    The pre-fix bug: pool_num/pool_den already track per-position
    conditionals, then _seed() passed them through
    unconditional_to_conditional_rates, distorting the values (e.g.
    true [0.9, 0.8] became ~[0.9, 0.889]).
    """
    est = AcceptanceHistoryEstimator(1, 2, alpha=1.0, min_count=1, warmup_steps=1)
    # 10 steps at width 2: position 1 always verified; position 2 verified
    # when pos-1 accepted. Feed known outcomes to produce a known pool.
    for _ in range(10):
        est.update(0, 2, 2)  # always fully accepted
    # pool for pos 1: 10/10 = 1.0; pos 2: 10/10 = 1.0
    seed = est._seed()
    assert seed[0] == pytest.approx(1.0)
    assert seed[1] == pytest.approx(1.0)

    # Now with partial acceptance
    est = AcceptanceHistoryEstimator(1, 2, alpha=1.0, min_count=1, warmup_steps=1)
    for _ in range(10):
        est.update(0, 2, 1)  # always accepted exactly 1 (pos 2 rejected)
    # pool pos 1: 10 accepts / 10 verified = 1.0
    # pool pos 2: 0 accepts / 10 verified-and-reached = 0.0
    seed = est._seed()
    assert seed[0] == pytest.approx(1.0)
    assert seed[1] == pytest.approx(0.0)


def test_seed_mixed_acceptance_no_double_conversion():
    """With a 50% pool, the seed must be 0.5 — not 0.5/0.5 = 1.0."""
    est = AcceptanceHistoryEstimator(1, 2, alpha=1.0, min_count=1, warmup_steps=1)
    for i in range(20):
        est.update(0, 2, 2 if i % 2 == 0 else 1)
    seed = est._seed()
    assert seed[0] == pytest.approx(1.0)  # always accept at pos 1
    assert seed[1] == pytest.approx(0.5)  # half accept at pos 2


def test_per_position_shrinkage():
    """Sparsely observed tails must not inherit head confidence.

    The pre-fix bug: shrinkage used the request's TOTAL step count for
    every position, so a position observed once got the confidence of
    a position observed 100 times.
    """
    est = AcceptanceHistoryEstimator(1, 4, alpha=0.3, min_count=8, warmup_steps=1)
    # 100 steps at width 2, but only 2 steps at width 4
    for _ in range(98):
        est.update(0, 2, 2)
    est.update(0, 4, 4)  # pos 4 observed (accepted 4)
    est.update(0, 4, 3)  # pos 4 observed (reached 3, rejected at 4)
    counts = est._pos_counts[0]
    assert counts[0] >= 98  # pos 1: nearly every step
    assert counts[3] == 2  # pos 4: only 2 observations
    # The shrinkage confidence must differ
    cond = est.conditionals(np.array([0]))[0]
    # pos 1 should lean toward its own EMA (high confidence)
    # pos 4 should lean toward the seed (low confidence from 2 obs)
    assert cond[0] != cond[3]  # different shrinkage weights
