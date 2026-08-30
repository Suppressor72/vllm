# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Adaptive verification for DSpark speculative decoding."""

import os
from collections import defaultdict
from collections.abc import Iterable, Iterator
from typing import TYPE_CHECKING

import numpy as np
import torch

import vllm.envs as envs
from vllm.distributed.parallel_state import get_tp_group
from vllm.logger import init_logger
from vllm.utils.gpu_sync_debug import gpu_sync_allowed
from vllm.v1.attention.backend import AttentionCGSupport
from vllm.v1.utils import CpuGpuBuffer
from vllm.v1.worker.gpu.async_utils import StepTimingSample, stream
from vllm.v1.worker.gpu.attn_utils import (
    get_attn_cg_support,
    get_query_lens_mismatch_unsupported_backend,
)
from vllm.v1.worker.gpu.buffer_utils import async_copy_to_gpu

logger = init_logger(__name__)
_PROFILE_REPLAYS = 5

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.worker.gpu.attn_utils import AttentionCGSupportInfo
    from vllm.v1.worker.gpu.input_batch import InputBatch
    from vllm.v1.worker.gpu.states import RequestState
    from vllm.v1.worker.utils import AttentionGroup


def _assign_draft_token_budget(
    confidence_probs: torch.Tensor,
    idx_mapping: torch.Tensor,
    capacities: torch.Tensor,
    draft_budget: int,
    num_steps: int,
    reserved: torch.Tensor | None = None,
    reserved_total: int = 0,
) -> None:
    """Admit the globally best `draft_budget` draft slots, in place.

    Every (request, step) slot is scored by its survival probability, the running
    product of that request's per-position confidences, and the highest scores win.
    Survival only decreases along a request, so a global top-k always admits
    continuously along steps with a request.

    When `reserved` carries per-request reserved counts (total `reserved_total`,
    both host-known), each request's first reserved[i] positions — exactly its
    best — are pre-admitted and the top-k fills only the remainder of the budget.
    With reserved_total == 0 the original pure top-k runs unchanged.

    `capacities` enters holding each request's scheduled draft count (which bounds its
    eligible slots) and leaves holding the admitted count. The caller only calls this
    when `draft_budget < capacities.sum()`, so every winner is a real draft slot.
    """
    survival = confidence_probs[idx_mapping].cumprod(dim=1)
    steps = torch.arange(num_steps, device=survival.device)
    # Out-of-range slots score -inf so they never outrank a real draft.
    out_of_range = steps[None, :] >= capacities[:, None]
    survival = survival.masked_fill(out_of_range, -float("inf"))
    flat = survival.flatten()
    if reserved_total > 0:
        assert reserved is not None
        reserved_mask = steps[None, :] < reserved[:, None]
        reserved_mask &= ~out_of_range
        # Reserved slots are already admitted; rank only the rest.
        scores = flat.masked_fill(reserved_mask.flatten(), -float("inf"))
        remaining = draft_budget - reserved_total
        admitted = reserved_mask.flatten().clone()
        if remaining > 0:
            winners = scores.topk(remaining).indices
            admitted.index_fill_(0, winners, True)
    else:
        winners = flat.topk(draft_budget).indices
        admitted = torch.zeros_like(flat, dtype=torch.bool).index_fill_(
            0, winners, True
        )
    torch.sum(admitted.view_as(survival), dim=1, dtype=capacities.dtype, out=capacities)


_assign_draft_token_budget_compiled = torch.compile(
    _assign_draft_token_budget, dynamic=True
)


def _min_width_reservation(
    scheduled_drafts: np.ndarray,
    min_width: int,
    max_verifying_reqs: int,
    max_draft_budget: int,
) -> np.ndarray:
    """Per-request reserved draft counts for the concurrency-gated width
    floor (all zeros when the floor is off).

    The floor is active only while the number of VERIFYING requests
    (scheduled drafts > 0) is at most `max_verifying_reqs`; each such
    request reserves min(min_width, its scheduled count). Non-verifying
    rows always reserve zero. Under the sampler logits-chunk cap the
    reservation is best-effort: each request's reserve scales down to
    ``max_draft_budget // num_verifying``.
    """
    num_verifying = int((scheduled_drafts > 0).sum())
    if min_width <= 0 or num_verifying == 0 or num_verifying > max_verifying_reqs:
        return np.zeros_like(scheduled_drafts)
    reserved = np.minimum(scheduled_drafts, min_width)
    total = int(reserved.sum())
    if total > max_draft_budget:
        per_request_cap = max_draft_budget // num_verifying
        reserved = np.minimum(reserved, per_request_cap)
    return reserved


def _history_env(name: str, default: str) -> str:
    return os.environ.get(name, default)


class AcceptanceHistoryEstimator:
    """Censoring-aware per-request acceptance history (history mode).

    Tracks, per request slot, a filtered EMA of ind(accepted >= p) over
    steps that REACHED p-1 with position p actually verified — the
    Kaplan-Meier-style predicate ``admitted >= p AND accepted >= p-1``.
    A step that died before p-1, or whose death coincided with the trim
    boundary, simply skips that position's update: unreached is not
    rejected. Chunked-prefill rows are skipped entirely upstream (their
    ``num_rejected`` is force-zeroed by the sampling kernel).

    The stored values are CONDITIONAL per-position probabilities
    q_p = P(accept_p | reached p-1): both budget consumers cumprod their
    inputs into survival, so storing survival indicators would
    double-compose. A pooled CONDITIONAL histogram (the pool counts are
    already per-position conditionals — never converted a second time)
    seeds cold requests; shrinkage is PER-POSITION (each position blends
    toward the seed by its own observation count, so sparsely observed
    tails stay seed-dominated). An empty pool seeds q=1.0
    (full-width-biased, matching the per-request warmup pin). Delayed
    D2H outcomes are mapped against the PRODUCING step's slot buffer,
    so batch reorders between steps cannot misattribute observations.
    """

    def __init__(
        self,
        num_slots: int,
        num_steps: int,
        alpha: float,
        min_count: int,
        warmup_steps: int,
    ) -> None:
        self.num_steps = num_steps
        self.alpha = alpha
        self.min_count = min_count
        self.warmup_steps = warmup_steps
        self._ema = np.ones((num_slots, num_steps), dtype=np.float64)
        self._steps = np.zeros(num_slots, dtype=np.int64)
        # Per-(slot, position) counts: shrinkage weight per position.
        self._pos_counts = np.zeros((num_slots, num_steps), dtype=np.int64)
        # Pooled CONDITIONAL ladder: P(accept_p | reached p-1) over all
        # request-steps where position p was verified and reached.
        self._pool_num = np.zeros(num_steps, dtype=np.int64)
        self._pool_den = np.zeros(num_steps, dtype=np.int64)

    def reset(self, slot: int) -> None:
        self._ema[slot].fill(1.0)
        self._steps[slot] = 0
        self._pos_counts[slot].fill(0)

    def update(self, slot: int, admitted: int, accepted: int) -> None:
        """Apply one (admitted, accepted) observation for a request.

        `admitted` is the verify width that actually ran; `accepted`
        the surviving prefix length (accepted <= admitted).
        """
        if admitted <= 0:
            return
        self._steps[slot] += 1
        for p in range(1, self.num_steps + 1):
            if admitted >= p and accepted >= p - 1:
                survived = 1.0 if accepted >= p else 0.0
                ema = self._ema[slot, p - 1]
                self._ema[slot, p - 1] = (1.0 - self.alpha) * ema + self.alpha * (
                    survived
                )
                self._pool_num[p - 1] += int(accepted >= p)
                self._pool_den[p - 1] += 1
                self._pos_counts[slot, p - 1] += 1

    def _seed(self) -> np.ndarray:
        """Pooled conditional ladder as the seed (q=1 if empty pool).

        The pool counts are already per-position conditionals (updates
        fire only when position p was verified AND reached p-1), so
        they seed directly — no second conversion. Unobserved positions
        seed optimistically (q=1): censored, not zero.
        """
        if (self._pool_den > 0).any():
            return np.divide(
                self._pool_num,
                self._pool_den,
                out=np.ones(self.num_steps, dtype=np.float64),
                where=self._pool_den > 0,
            ).clip(0.0, 1.0)
        return np.ones(self.num_steps, dtype=np.float64)

    def conditionals(self, slots: np.ndarray) -> np.ndarray:
        """Shrunk conditional q_p for the given slots, shape (len(slots), K).

        Shrinkage is per-position: each position blends toward the seed
        by its own observation count, so a sparsely observed tail leans
        on the pooled seed while a well-observed head trusts its EMA.
        """
        seed = self._seed()
        counts = self._pos_counts[slots].astype(np.float64)
        confidence = counts / (counts + self.min_count)
        raw = self._ema[slots]
        return (confidence * raw + (1.0 - confidence) * seed[None, :]).clip(0.0, 1.0)

    def is_cold(self, slots: np.ndarray) -> np.ndarray:
        return self._steps[slots] < self.warmup_steps


def build_cost_tables_from_curves(
    draft_curve: list[tuple[int, float]],
    verify_curve: list[tuple[int, float]],
    max_num_reqs: int,
    max_batch_tokens: int,
    cudagraph_limit: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Build cost tables: graph-padded below the capture limit, smooth above.

    Args:
        cudagraph_limit: Largest cudagraph-captured size. At or below it,
            execution pads up to the next captured size, so cost is a step
            function. Above it there is no padding, so cost is continuous.
    """

    def build_table(limit: int, curve: list[tuple[int, float]]) -> np.ndarray:
        xs, ys = np.asarray(curve, dtype=np.float64).T
        ys = np.maximum.accumulate(ys)
        values = np.arange(limit + 1)
        # Execution pads to the next captured size, so cost is a step
        # function of the padded size: smooth interpolation would invent
        # marginal per-token costs that don't exist within a pad bucket.
        idx = np.searchsorted(xs, values, side="left")
        result = ys[np.minimum(idx, len(xs) - 1)]
        # Past the capture limit nothing pads, so cost really is continuous in
        # size; snapping to the next profiled point overestimates badly when
        # the profiled points are far apart. Interpolate only between points
        # that are themselves past the limit: crossing the limit loses
        # cudagraphs entirely, which is a genuine discontinuity, so the first
        # point above it must not be blended with the last one below.
        if cudagraph_limit:
            smooth = values > cudagraph_limit
            above = xs > cudagraph_limit
            if smooth.any() and above.any():
                result[smooth] = np.interp(values[smooth], xs[above], ys[above])
        if len(xs) > 1:
            after = values > xs[-1]
            slope = (ys[-1] - ys[-2]) / (xs[-1] - xs[-2])
            result[after] = ys[-1] + slope * (values[after] - xs[-1])
        return result

    draft_table = np.maximum(build_table(max_num_reqs, draft_curve), 0.0)
    verify_table = np.maximum(build_table(max_batch_tokens, verify_curve), 1e-6)
    return draft_table, verify_table


class AdaptiveVerificationManager:
    def __init__(
        self,
        req_states: "RequestState",
        query_start_loc: torch.Tensor,
        num_bonus_tokens: int,
        max_total_logits: int,
        min_draft_width: int = 0,
        min_width_max_reqs: int = 2,
        confidence_source: str = "head",
    ):
        self.req_states = req_states
        self.num_speculative_steps = req_states.num_speculative_steps
        device = req_states.device
        self._copy_stream = torch.cuda.Stream(device)

        self.num_bonus_tokens = num_bonus_tokens
        # Concurrency-gated width floor (0 disables; see
        # _min_width_reservation for the exact semantics).
        self.min_draft_width = min_draft_width
        self.min_width_max_reqs = min_width_max_reqs
        self._floor_cap_warned = False
        # "head": the speculator publishes per-step confidence probs
        # (DSpark). "history": the manager derives censoring-aware
        # per-request conditionals from observed acceptance (DFlash2,
        # which has no confidence head).
        self.confidence_source = confidence_source
        self._history: AcceptanceHistoryEstimator | None = None
        self._hist_bufs: list[CpuGpuBuffer] | None = None
        self._hist_events: list[torch.cuda.Event] | None = None
        self._hist_idx = 0
        self._hist_pending_resets: list[int] = []
        if confidence_source == "history":
            self._history = AcceptanceHistoryEstimator(
                req_states.max_num_reqs,
                self.num_speculative_steps,
                alpha=float(_history_env("VLLM_ADAPTIVE_HISTORY_ALPHA", "0.15")),
                min_count=int(_history_env("VLLM_ADAPTIVE_HISTORY_MIN_COUNT", "8")),
                warmup_steps=int(
                    _history_env("VLLM_ADAPTIVE_HISTORY_WARMUP_STEPS", "8")
                ),
            )
            # Dedicated int32 D2H slots: rows 0/1 = (num_rejected,
            # admitted_width) per batch row; row 2 = the producing
            # step's slot per row (delayed updates map by THIS, not the
            # reader's mapping). Never packed into float32 matrices.
            self._hist_bufs = [
                CpuGpuBuffer(
                    3,
                    req_states.max_num_reqs,
                    dtype=torch.int32,
                    device=device,
                )
                for _ in range(2)
            ]
            self._hist_events = [torch.cuda.Event(blocking=True) for _ in range(2)]
        # Rejection sampling verifies logits in one contiguous chunk; the
        # chunked path indexes by scheduled (untrimmed) offsets and cannot
        # address the compacted layout, so the budget must fit one chunk.
        self._max_total_logits = max_total_logits
        self.query_start_loc = query_start_loc
        self.cost_tables: tuple[np.ndarray, np.ndarray] | None = None
        # Largest cudagraph-captured token count; above it nothing pads.
        self._cudagraph_limit = 0
        # (drafts/req, non-draft/req, budget, reserved/req)
        self._batch_budget: (
            tuple[dict[str, int], dict[str, int], int, dict[str, int]] | None
        ) = None
        max_num_reqs = req_states.max_num_reqs
        # Current per-slot confidences
        self._confidence_probs = torch.empty(
            (max_num_reqs, self.num_speculative_steps),
            dtype=torch.float32,
            device=device,
        )
        self._batch_draft_capacity = torch.empty(
            max_num_reqs, dtype=torch.int32, device=device
        )
        self._reserved_drafts = torch.empty(
            max_num_reqs, dtype=torch.int32, device=device
        )
        self._num_non_draft_tokens = torch.empty_like(query_start_loc[:-1])
        self._cu_num_logits = torch.empty_like(query_start_loc)
        # Exploration/audit arm (benchmark/debug ONLY, default off):
        # with probability _audit_pct, an eligible trimmed step verifies
        # FULL width instead, yielding uncensored per-slot observations.
        # Fixed seed: TP replicas execute the identical call sequence,
        # so both ranks draw the same audit steps and stay lockstep.
        self._audit_pct = float(_history_env("VLLM_ADAPTIVE_AUDIT_FULL_WIDTH_PCT", "0"))
        self._audit_rng = np.random.default_rng(0x5EED)

        # Two D2H slots preserve stale inputs for budget selection.
        self._stale_confidences = [
            CpuGpuBuffer(
                max_num_reqs,
                self.num_speculative_steps,
                dtype=torch.float32,
                device=device,
            )
            for _ in range(2)
        ]
        self._copy_events = [torch.cuda.Event(blocking=True) for _ in range(2)]
        self._pending_resets: list[int] = []
        self._stale_idx = 0
        for slot in self._stale_confidences:
            slot.np.fill(1.0)

    def add_request(self, req_idx: int) -> None:
        self._stale_confidences[self._stale_idx].np[req_idx].fill(1.0)
        self._pending_resets.append(req_idx)
        self._confidence_probs[req_idx].fill_(1.0)
        if self._history is not None:
            self._history.reset(req_idx)
            self._hist_pending_resets.append(req_idx)

    def batches_to_profile(self, capture_sizes: list[int]) -> Iterator[dict[str, int]]:
        """Dummy-run kwargs whose step timings seed the cost tables.

        Run these inside StepTimingCollector.collect(), then hand the block's
        timings to set_initial_cost_curves."""
        max_num_tokens = self.req_states.max_num_batched_tokens
        size = self._cudagraph_limit = capture_sizes[-1] if capture_sizes else 0
        # Also profile beyond the capture limit: real steps run there
        # (piecewise/eager) and linear extrapolation badly underestimates
        # them. These runs double as JIT warmup for the piecewise shapes.
        tail_sizes: set[int] = set()
        if size:
            tail_sizes.add(min(size + size // 2, max_num_tokens))
            while size < max_num_tokens:
                size = min(size * 2, max_num_tokens)
                tail_sizes.add(size)
            tail_sizes -= set(capture_sizes)
        for num_tokens in capture_sizes + sorted(tail_sizes):
            for _ in range(_PROFILE_REPLAYS):
                yield {
                    "num_tokens": num_tokens,
                    "context_len": envs.VLLM_ADAPTIVE_VERIFICATION_PROFILE_CONTEXT_LEN,
                }

    def set_initial_cost_curves(self, samples: list[StepTimingSample]) -> None:
        def median_curve(
            points: Iterable[tuple[int, float]],
        ) -> list[tuple[int, float]]:
            grouped: defaultdict[int, list[float]] = defaultdict(list)
            for key, value in points:
                grouped[key].append(value)
            return [(k, float(np.median(v))) for k, v in sorted(grouped.items())]

        # Draft curve: eager-target steps inflate drafter timings (the CPU is
        # still launching kernels, opening gaps between the drafter's events),
        # and request counts — unlike token counts — collide across execution
        # modes, so only graph-replay samples may price the draft curve.
        draft_curve = median_curve(
            (s.num_reqs, s.drafter_ms) for s in samples if s.full_cudagraph
        )
        verify_curve = median_curve(
            (s.num_target_tokens, s.forward_ms) for s in samples
        )
        self.set_cost_curves(draft_curve, verify_curve)

    def set_cost_curves(
        self,
        draft_curve: list[tuple[int, float]],
        verify_curve: list[tuple[int, float]],
    ) -> None:
        draft_curve, verify_curve = get_tp_group().broadcast_object(
            (draft_curve, verify_curve), src=0
        )
        if not draft_curve or not verify_curve:
            raise RuntimeError(
                "Adaptive verification could not profile step costs. Pass "
                "`enable_adaptive_verification=false` in the speculative config to "
                "verify a fixed number of drafts instead."
            )
        self.cost_tables = build_cost_tables_from_curves(
            draft_curve,
            verify_curve,
            self.req_states.max_num_reqs,
            self.req_states.max_num_batched_tokens,
            self._cudagraph_limit,
        )
        logger.debug("DSpark cost tables: %s", self.cost_tables)

    def record_confidences(
        self,
        confidence_probs: torch.Tensor,
        input_batch: "InputBatch",
    ) -> None:
        """Publish this step's raw confidences for the ranking kernel and start
        copying them to the CPU, where a later step's budget reads them."""
        num_reqs = input_batch.num_reqs
        ready_idx = self._stale_idx ^ 1
        with gpu_sync_allowed():
            self._copy_events[ready_idx].synchronize()
        if self._pending_resets:
            self._stale_confidences[ready_idx].np[self._pending_resets] = 1.0
            self._pending_resets.clear()
        # Last step's copy has landed: budgets read it, this step overwrites the
        # slot they were reading before.
        self._stale_idx, write_idx = ready_idx, self._stale_idx

        self._confidence_probs[input_batch.idx_mapping] = confidence_probs[:num_reqs]
        write_slot = self._stale_confidences[write_idx]
        write_slot.gpu.copy_(self._confidence_probs)

        current_stream = torch.cuda.current_stream(self.req_states.device)
        self._copy_stream.wait_stream(current_stream)
        with stream(self._copy_stream, current_stream):
            write_slot.copy_to_cpu()
            self._copy_events[write_idx].record()

    def record_acceptance(
        self,
        num_rejected: torch.Tensor,
        input_batch: "InputBatch",
    ) -> None:
        """History mode: ingest this step's per-row verify outcomes.

        `num_rejected` (device, batch-row order) pairs with the admitted
        widths the allocator chose this step (`_batch_draft_capacity`,
        still valid post-verify). One (num_rejected, admitted) int32 pair
        per row goes to the CPU through the same double-buffer + event +
        side-stream discipline as the head confidences; the landed
        previous-step pair feeds the estimator below. Rows still in
        chunked prefill are skipped — the sampling kernel zeroes their
        num_rejected, which would read as fake full-accepts.
        """
        assert self._history is not None
        assert self._hist_bufs is not None
        assert self._hist_events is not None
        hist_bufs = self._hist_bufs
        hist_events = self._hist_events
        num_reqs = input_batch.num_reqs
        ready_idx = self._hist_idx ^ 1
        with gpu_sync_allowed():
            hist_events[ready_idx].synchronize()
        landed = hist_bufs[ready_idx]
        if self._hist_pending_resets:
            for slot in self._hist_pending_resets:
                self._history.reset(slot)
            self._hist_pending_resets.clear()
        # The landed slot holds the PREVIOUS step's outcomes AND that
        # step's slot mapping (row 2, written at copy time): delayed
        # updates map against the PRODUCING step's slots, so a batch
        # reorder between steps cannot misattribute observations.
        producer_slots = landed.np[2, :num_reqs]
        rejected_np = landed.np[0, :num_reqs]
        admitted_np = landed.np[1, :num_reqs]
        for row in range(num_reqs):
            slot = int(producer_slots[row])
            admitted = int(admitted_np[row])
            accepted = admitted - int(rejected_np[row])
            self._history.update(slot, admitted, accepted)
        # Rotate: read the landed slot next time, write this step into
        # the slot nothing reads.
        self._hist_idx, write_idx = ready_idx, self._hist_idx
        write_slot = hist_bufs[write_idx]
        write_slot.gpu[0, :num_reqs].copy_(num_rejected[:num_reqs])
        write_slot.gpu[1, :num_reqs].copy_(self._batch_draft_capacity[:num_reqs])
        # Buffer the producing step's slot mapping for the delayed read.
        write_slot.gpu[2, :num_reqs].copy_(input_batch.idx_mapping[:num_reqs])
        current_stream = torch.cuda.current_stream(self.req_states.device)
        self._copy_stream.wait_stream(current_stream)
        with stream(self._copy_stream, current_stream):
            write_slot.copy_to_cpu()
            hist_events[write_idx].record()

    def get_num_tokens(
        self,
        num_tokens_per_req: dict[str, int],
        draft_tokens: dict[str, list[int]],
    ) -> int:
        """Token count once the draft budget is trimmed to fit.

        Stashes the chosen budget in ``_batch_budget`` for the compaction and
        reallocation that follow in the same step.
        """
        assert self.cost_tables is not None
        req_ids = list(num_tokens_per_req)
        num_reqs = len(req_ids)
        scheduled_tokens = np.fromiter(
            num_tokens_per_req.values(), dtype=np.int32, count=num_reqs
        )
        scheduled_drafts = np.fromiter(
            (len(draft_tokens.get(req_id, ())) for req_id in req_ids),
            dtype=np.int32,
            count=num_reqs,
        )
        num_non_draft_tokens = scheduled_tokens - scheduled_drafts
        slots = np.fromiter(
            (self.req_states.req_id_to_index[req_id] for req_id in req_ids),
            dtype=np.int32,
            count=len(req_ids),
        )
        if self._history is not None:
            stale_confidences = self._history.conditionals(slots)
            # Mirror the conditionals into the GPU rank kernel's buffer
            # (head mode refreshes it per record_confidences; history
            # mode refreshes it at budget time). Advanced indexing does
            # not cross devices implicitly, so move both sides.
            device = self._confidence_probs.device
            slots_t = torch.from_numpy(slots.astype(np.int64)).to(device)
            self._confidence_probs[slots_t] = torch.from_numpy(
                stale_confidences.astype(np.float32)
            ).to(device)
        else:
            stale_confidences = self._stale_confidences[self._stale_idx].np[slots]
        survival_probability = np.cumprod(stale_confidences.astype(np.float64), axis=1)
        steps = np.arange(self.num_speculative_steps)
        valid = steps[None, :] < scheduled_drafts[:, None]
        scores = np.sort(survival_probability[valid])[::-1]
        num_non_draft_tokens_total = int(num_non_draft_tokens.sum())
        max_draft_budget = min(
            int(scheduled_drafts.sum()),
            max(0, self._max_total_logits - num_reqs * self.num_bonus_tokens),
        )
        scores = scores[:max_draft_budget]
        draft_cost_ms, verify_cost_ms = self.cost_tables
        num_sampling_requests = np.count_nonzero(
            self.req_states.num_computed_tokens_np[slots] + num_non_draft_tokens
            >= self.req_states.prefill_len.np[slots]
        )
        num_tokens_to_estimated_accepted_tokens = np.concatenate(
            ([num_sampling_requests], num_sampling_requests + np.cumsum(scores))
        )
        costs = (
            draft_cost_ms[len(req_ids)]
            + verify_cost_ms[
                num_non_draft_tokens_total : num_non_draft_tokens_total
                + max_draft_budget
                + 1
            ]
        )
        num_drafts_per_req = {
            req_id: int(num_drafts)
            for req_id, num_drafts in zip(req_ids, scheduled_drafts, strict=True)
        }
        num_non_draft_tokens_per_req = {
            req_id: int(num_tokens)
            for req_id, num_tokens in zip(req_ids, num_non_draft_tokens, strict=True)
        }
        argmax_budget = int(np.argmax(num_tokens_to_estimated_accepted_tokens / costs))
        # ONE mode-split reservation vector (a single code path):
        #   floor term — active while verifying requests <= max_reqs
        #     (mode-independent);
        #   history-only terms — cold-request warmup pin (scheduled
        #     width) and the min-1 starvation guard for verifying rows
        #     (censoring is what makes admitted=0 absorbing; the head
        #     keeps emitting scores after a 0-width step, so head mode
        #     reserves nothing and stays byte-identical at F=0).
        floor_res = _min_width_reservation(
            scheduled_drafts,
            self.min_draft_width,
            self.min_width_max_reqs,
            max_draft_budget,
        )
        if self._history is not None:
            cold = self._history.is_cold(slots)
            history_res = np.where(
                cold, scheduled_drafts, (scheduled_drafts > 0).astype(np.int32)
            )
        else:
            history_res = np.zeros_like(scheduled_drafts)
        reserved = np.minimum(scheduled_drafts, np.maximum(floor_res, history_res))
        reserved_total = int(reserved.sum())
        if (
            reserved_total > 0
            and int(np.minimum(scheduled_drafts, self.min_draft_width).sum())
            > max_draft_budget
            and not self._floor_cap_warned
        ):
            self._floor_cap_warned = True
            logger.warning(
                "Adaptive width floor scaled down: reservations %d exceed the "
                "logits-chunk budget %d; per-request reserves capped at %d.",
                int(np.minimum(scheduled_drafts, self.min_draft_width).sum()),
                max_draft_budget,
                max(0, max_draft_budget // max(1, int((scheduled_drafts > 0).sum()))),
            )
        draft_budget = max(argmax_budget, min(max_draft_budget, reserved_total))
        # Audit arm: occasionally override the trim with full width so
        # censored positions get observed. Eligible = a verifying step
        # the policy actually trimmed; each audit step is logged for the
        # policy-vs-audit estimator comparison.
        scheduled_total = int(scheduled_drafts.sum())
        if (
            self._audit_pct > 0
            and scheduled_total > 0
            and draft_budget < min(scheduled_total, max_draft_budget)
            and self._audit_rng.random() < self._audit_pct
        ):
            forced = min(scheduled_total, max_draft_budget)
            logger.info(
                "ADAPTIVE-AUDIT-FULL-WIDTH reqs=%d policy_budget=%d forced=%d",
                num_reqs,
                draft_budget,
                forced,
            )
            draft_budget = forced
        self._batch_budget = (
            num_drafts_per_req,
            num_non_draft_tokens_per_req,
            draft_budget,
            {req_id: int(r) for req_id, r in zip(req_ids, reserved, strict=True)},
        )
        return sum(num_non_draft_tokens_per_req.values()) + draft_budget

    def compact_batch(
        self,
        num_draft_tokens_per_req: np.ndarray,
        num_scheduled_tokens: np.ndarray,
        cu_num_logits_np: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compact the CPU batch to the chosen draft budget.

        Returns the compacted per-request token counts and the CPU cu_num_logits_np.
        If the draft budget is 0, we can know cu_num_logits_np exactly, otherwise
        its unchanged/an-upper-bound.
        """
        batch_budget = self._batch_budget
        assert batch_budget is not None
        _, _, draft_budget, _reserved = batch_budget
        num_drafts = int(num_draft_tokens_per_req.sum())
        if draft_budget == num_drafts:
            return num_scheduled_tokens, cu_num_logits_np

        num_non_draft_tokens = num_scheduled_tokens - num_draft_tokens_per_req
        if draft_budget == 0:
            # The draft budget is 0, so we can know cu_num_logits_np exactly. This helps
            # when we would exceed the sampler logit chunk size.
            num_reqs = num_scheduled_tokens.shape[0]
            cu_num_logits_np = (
                np.arange(num_reqs + 1, dtype=cu_num_logits_np.dtype)
                * self.num_bonus_tokens
            )
            return num_non_draft_tokens, cu_num_logits_np

        is_verification_request = num_draft_tokens_per_req > 0
        num_verification_reqs = int(is_verification_request.sum())
        # sort_batch_req_ids keeps verification requests at the front.
        assert np.all(is_verification_request[:num_verification_reqs])
        # for the CPU side buffer we distribute draft tokens evenly
        draft_lens_cpu = np.zeros_like(num_non_draft_tokens)
        draft_lens_cpu[:num_verification_reqs] = draft_budget // num_verification_reqs
        draft_lens_cpu[: draft_budget % num_verification_reqs] += 1
        return num_non_draft_tokens + draft_lens_cpu, cu_num_logits_np

    def reallocate_drafts(
        self, req_ids: list[str], idx_mapping: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        batch_budget, self._batch_budget = self._batch_budget, None
        assert batch_budget is not None
        (
            num_drafts_per_req,
            num_non_draft_tokens_per_req,
            draft_budget,
            reserved_per_req,
        ) = batch_budget
        num_reqs = idx_mapping.shape[0]
        scheduled_drafts = np.fromiter(
            (num_drafts_per_req[req_id] for req_id in req_ids),
            dtype=np.int32,
            count=num_reqs,
        )
        num_non_draft_tokens = np.fromiter(
            (num_non_draft_tokens_per_req[req_id] for req_id in req_ids),
            dtype=np.int32,
            count=num_reqs,
        )
        num_tokens = int(num_non_draft_tokens.sum()) + draft_budget

        # Rank draft slots by survival probability and admit the best prefix.
        # capacities enters holding each request's valid draft count (the kernel
        # uses it to bound eligible slots) and leaves holding the admitted count.
        capacities = self._batch_draft_capacity[:num_reqs]
        if draft_budget == 0:
            capacities.zero_()
        else:
            async_copy_to_gpu(scheduled_drafts, out=capacities)
            if draft_budget < int(scheduled_drafts.sum()):
                reserved_np = np.fromiter(
                    (reserved_per_req.get(req_id, 0) for req_id in req_ids),
                    dtype=np.int32,
                    count=num_reqs,
                )
                reserved_total = int(reserved_np.sum())
                reserved_gpu = None
                if reserved_total > 0:
                    reserved_gpu = self._reserved_drafts[:num_reqs]
                    async_copy_to_gpu(reserved_np, out=reserved_gpu)
                _assign_draft_token_budget_compiled(
                    self._confidence_probs,
                    idx_mapping,
                    capacities,
                    draft_budget,
                    self.num_speculative_steps,
                    reserved_gpu,
                    reserved_total,
                )

        num_non_draft_tokens_gpu = self._num_non_draft_tokens[:num_reqs]
        async_copy_to_gpu(
            num_non_draft_tokens,
            out=num_non_draft_tokens_gpu,
        )
        self._cu_num_logits[:1].zero_()
        torch.cumsum(
            capacities + self.num_bonus_tokens,
            dim=0,
            out=self._cu_num_logits[1 : num_reqs + 1],
        )
        self.query_start_loc[:1].zero_()
        torch.cumsum(
            capacities + num_non_draft_tokens_gpu,
            dim=0,
            out=self.query_start_loc[1 : num_reqs + 1],
        )
        self.query_start_loc[num_reqs + 1 :].fill_(num_tokens)
        return (
            self._cu_num_logits[: num_reqs + 1],
            self.query_start_loc,
            draft_budget,
        )


def maybe_create_adaptive_verification_manager(
    *,
    enable_adaptive_verification: bool,
    attn_groups: list[list["AttentionGroup"]],
    attn_cg_support: "AttentionCGSupportInfo",
    req_states: "RequestState",
    query_start_loc: torch.Tensor,
    num_bonus_tokens: int,
    max_total_logits: int,
    vllm_config: "VllmConfig",
    target_layer_names: set[str] | None = None,
    additional_attn_cg_support: tuple[AttentionCGSupport, str | None] | None = None,
    confidence_source: str = "head",
) -> AdaptiveVerificationManager | None:
    if not enable_adaptive_verification:
        return None

    # The selector rejects unsupported backends, but models that
    # hard-wire theirs (e.g. DeepSeek-V4) never go through it.
    backend = get_query_lens_mismatch_unsupported_backend(
        attn_groups,
        checked_layer_names=target_layer_names,
    )
    if backend is not None:
        raise ValueError(
            "Adaptive verification trims verification requests on device, which"
            f" the {backend} attention backend does not support. Pass "
            "enable_adaptive_verification=false in the speculative config, or "
            "use a backend that does."
        )

    target_attn_cg_support = attn_cg_support
    if target_layer_names is not None:
        target_attn_cg_support = get_attn_cg_support(
            attn_groups,
            vllm_config,
            checked_layer_names=target_layer_names,
        )
        if additional_attn_cg_support is not None:
            target_attn_cg_support = target_attn_cg_support.narrow(
                *additional_attn_cg_support
            )
    if target_attn_cg_support.min_cg_support not in (
        AttentionCGSupport.VARLEN_DECODE,
        AttentionCGSupport.ALWAYS,
    ):
        raise ValueError(
            "Adaptive verification captures varlen decode cudagraphs, so every"
            " target attention builder must report "
            "AttentionCGSupport.VARLEN_DECODE or AttentionCGSupport.ALWAYS, but "
            f"{target_attn_cg_support.min_cg_attn_backend} reports "
            f"{target_attn_cg_support.min_cg_support}. Pass "
            "enable_adaptive_verification=false in the speculative config, or "
            "use a backend that does."
        )

    spec_config = vllm_config.speculative_config
    assert spec_config is not None
    return AdaptiveVerificationManager(
        req_states,
        query_start_loc,
        num_bonus_tokens,
        max_total_logits=max_total_logits,
        min_draft_width=getattr(spec_config, "adaptive_min_draft_width", 0),
        min_width_max_reqs=getattr(spec_config, "adaptive_min_width_max_reqs", 2),
        confidence_source=confidence_source,
    )
