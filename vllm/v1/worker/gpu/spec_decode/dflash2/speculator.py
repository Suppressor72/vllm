# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from typing import Any

import torch

from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.logger import init_logger
from vllm.triton_utils import tl, triton
from vllm.v1.worker.gpu.sample.gumbel import gumbel_noised_argmax
from vllm.v1.worker.gpu.spec_decode.dflash.speculator import DFlashSpeculator

logger = init_logger(__name__)

# Calibrated acceptance map for the selector provider: 20 quantile bins
# on raw centers (empty bins retained as plateaus), monotone via
# cumulative max (a weighted-PAVA refit scores within 0.002 AUC / 0.001
# Brier). Fit on the adaptive-dflash P2.2a-reliable training split
# (4,478 in-serving observations of realized selector scores and
# per-position acceptance; held-out Brier 0.133 / AUC 0.862). Input feature:
# softmax top-1 over the realized top-k
# selector scores = the BEST candidate's confidence (under greedy
# sampling the selected path token; under temperature the walk may
# differ — self-consistent with the calibration data, but estimates
# best-candidate confidence, not selected-token confidence). Disable
# with VLLM_ADAPTIVE_SELECTOR_CALIBRATION=0.
_SELECTOR_CAL_KNOTS = (
    0.2629,
    0.4122,
    0.4988,
    0.5702,
    0.6351,
    0.7004,
    0.7657,
    0.8271,
    0.8772,
    0.9179,
    0.9493,
    0.9694,
    0.9834,
    0.9925,
    0.9971,
    0.9991,
    0.9998,
    1.0,
    1.0,
    1.0,
)
_SELECTOR_CAL_VALS = (
    0.2589,
    0.2589,
    0.4196,
    0.4464,
    0.5357,
    0.5848,
    0.6368,
    0.6652,
    0.7054,
    0.7545,
    0.8296,
    0.8844,
    0.9367,
    0.9507,
    0.9862,
    0.9863,
    0.9863,
    0.9935,
    0.9935,
    0.9983,
)


def selector_acceptance_confidences(selector_scores: torch.Tensor) -> torch.Tensor:
    """Per-position acceptance estimate from realized selector scores.

    Softmax over the top-k realized scores, take the top-1 value: the
    probability mass the selector places on its best candidate at each
    speculative position (the selected path token is that candidate
    under greedy sampling).
    """
    return torch.softmax(selector_scores.float(), dim=-1).max(dim=-1).values


def apply_selector_calibration(
    p1: torch.Tensor, knots: torch.Tensor, vals: torch.Tensor
) -> torch.Tensor:
    """Piecewise-constant monotone map (right-continuous, clamped)."""
    idx = torch.searchsorted(knots, p1.contiguous(), right=True).clamp(1, len(knots))
    return vals[idx - 1]


@triton.jit
def _selector_walk_kernel(
    scores_ptr,
    candidate_ptr,
    sample_pos_ptr,
    req_state_ptr,
    temperature_ptr,
    seeds_ptr,
    tokens_ptr,
    realized_scores_ptr,
    num_steps: tl.constexpr,
    top_k: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SAMPLE_PROBABILISTIC: tl.constexpr,
    USE_FP64: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_K)
    mask = offsets < top_k
    req_state = tl.load(req_state_ptr + row * num_steps)
    valid = req_state >= 0
    temperature = tl.load(temperature_ptr + req_state, mask=valid, other=0.0)
    seed = tl.load(seeds_ptr + req_state, mask=valid, other=0)
    previous = 0
    for step in range(num_steps):
        flat = row * num_steps + step
        score_base = (flat * top_k + previous) * top_k
        scores = tl.load(
            scores_ptr + score_base + offsets,
            mask=mask & valid,
            other=float("-inf"),
        ).to(tl.float64 if USE_FP64 else tl.float32)
        candidate_base = flat * top_k
        candidates = tl.load(
            candidate_ptr + candidate_base + offsets,
            mask=mask & valid,
            other=0,
        )

        # Candidate ids key the noise, matching the target's own sampling.
        position = tl.load(sample_pos_ptr + flat) - 1
        _, index = gumbel_noised_argmax(
            scores,
            candidates,
            mask & valid,
            seed,
            position,
            temperature if SAMPLE_PROBABILISTIC else 0.0,
            USE_FP64=USE_FP64,
        )

        tl.store(
            realized_scores_ptr + candidate_base + offsets,
            scores,
            mask=mask & valid,
        )
        token = tl.load(candidate_ptr + candidate_base + index, mask=valid, other=0)
        tl.store(tokens_ptr + flat, token, mask=valid)
        previous = index


@triton.jit
def _cache_draft_logits_kernel(
    draft_logits_ptr,
    cached_candidate_ptr,
    candidate_ptr,
    scores_ptr,
    req_state_ptr,
    draft_logits_stride_0,
    draft_logits_stride_1,
    num_steps: tl.constexpr,
    top_k: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    flat = tl.program_id(0)
    req_state = tl.load(req_state_ptr + flat)
    step = flat % num_steps
    offsets = tl.arange(0, BLOCK_K)
    mask = (req_state >= 0) & (offsets < top_k)
    candidate_base = flat * top_k
    cache_base = (req_state * num_steps + step) * top_k
    old_token_ids = tl.load(cached_candidate_ptr + cache_base + offsets, mask=mask)
    logits_base = (
        draft_logits_ptr
        + req_state * draft_logits_stride_0
        + step * draft_logits_stride_1
    )
    tl.store(logits_base + old_token_ids, -float("inf"), mask=mask)
    token_ids = tl.load(candidate_ptr + candidate_base + offsets, mask=mask)
    scores = tl.load(scores_ptr + candidate_base + offsets, mask=mask)
    tl.store(logits_base + token_ids, scores, mask=mask)
    tl.store(cached_candidate_ptr + cache_base + offsets, token_ids, mask=mask)


class DFlash2Speculator(DFlashSpeculator):
    _speculator_name = "DFlash2"

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        super().__init__(vllm_config, device)
        # Adaptive verification feeds from observed-acceptance history:
        # DFlash2 checkpoints have no confidence head.
        self.enable_adaptive_verification = (
            self.speculative_config.enable_adaptive_verification
        )
        # Acceptance-estimate provider: selector by default (the P2.2
        # held-out winner), history selectable via config.
        self.adaptive_confidence_source = (
            self.speculative_config.adaptive_confidence_source or "selector"
        )
        if self.enable_adaptive_verification:
            self.draft_token_confidence_probs = torch.empty_like(
                self.draft_tokens, dtype=torch.float32
            )
            self._selector_cal_knots = torch.tensor(
                _SELECTOR_CAL_KNOTS, dtype=torch.float32, device=device
            )
            self._selector_cal_vals = torch.tensor(
                _SELECTOR_CAL_VALS, dtype=torch.float32, device=device
            )
            self._selector_calibrate = (
                os.environ.get("VLLM_ADAPTIVE_SELECTOR_CALIBRATION", "1") != "0"
            )
        if self.enable_adaptive_verification:
            logger.info(
                "DFlash2 adaptive acceptance provider: source=%s calibrated=%s",
                self.adaptive_confidence_source,
                self._selector_calibrate,
            )
        draft_config = self.draft_model_config.hf_config.dflash_config
        self.selector_top_k = int(draft_config["selector_top_k"])
        self._anchor_indices = (
            torch.arange(self.max_num_reqs, dtype=torch.int64, device=device)
            * self.num_query_per_req
        )
        self._selector_scores = torch.empty(
            self.max_num_reqs,
            self.num_speculative_steps,
            self.selector_top_k,
            dtype=torch.float32,
            device=device,
        )
        self._cached_candidate_ids = torch.zeros(
            self._selector_scores.shape, dtype=torch.int64, device=device
        )

    def draft_logits_spec(self, vllm_config: VllmConfig) -> tuple[torch.dtype, float]:
        # fp32 so the walk and the rejection that checks it read the same
        # distribution; -inf because the cache kernel writes only the K
        # candidates.
        return torch.float32, -float("inf")

    def _sample_path(
        self,
        candidate_ids: torch.Tensor,
        scores: torch.Tensor,
        num_reqs: int,
    ) -> None:
        block_k = triton.next_power_of_2(self.selector_top_k)
        _selector_walk_kernel[(num_reqs,)](
            scores.contiguous(),
            candidate_ids.contiguous(),
            self.sample_pos,
            self.sample_idx_mapping,
            self.temperature,
            self.seeds,
            self.draft_tokens,
            self._selector_scores,
            num_steps=self.num_speculative_steps,
            top_k=self.selector_top_k,
            BLOCK_K=block_k,
            SAMPLE_PROBABILISTIC=self.draft_logits is not None,
            USE_FP64=self.use_fp64_gumbel,
            num_warps=1,
        )

    def _cache_draft_logits(self, candidate_ids: torch.Tensor, num_sample: int) -> None:
        draft_logits = self.draft_logits
        assert draft_logits is not None
        block_k = triton.next_power_of_2(self.selector_top_k)
        _cache_draft_logits_kernel[(num_sample,)](
            draft_logits,
            self._cached_candidate_ids,
            candidate_ids,
            self._selector_scores,
            self.sample_idx_mapping,
            draft_logits.stride(0),
            draft_logits.stride(1),
            num_steps=self.num_speculative_steps,
            top_k=self.selector_top_k,
            BLOCK_K=block_k,
            num_warps=1,
        )

    def _generate_draft(
        self,
        num_reqs: int,
        num_tokens_padded: int,
        attn_metadata: dict[str, Any] | None,
        slot_mappings: dict[str, torch.Tensor] | None,
        num_tokens_across_dp: torch.Tensor | None,
        cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
    ) -> None:
        last_hidden_states = self._run_model(
            num_tokens_padded,
            attn_metadata,
            slot_mappings,
            num_tokens_across_dp,
            cudagraph_runtime_mode,
        )
        num_sample = num_reqs * self.num_speculative_steps
        hidden_states = last_hidden_states[self.sample_indices[:num_sample]].view(
            num_reqs, self.num_speculative_steps, -1
        )
        candidate_ids, unary_logits = self.model.compute_candidates(
            hidden_states.flatten(0, 1)
        )
        candidate_ids = candidate_ids.view(
            num_reqs, self.num_speculative_steps, self.selector_top_k
        )
        unary_logits = unary_logits.view_as(candidate_ids)
        anchor_token_ids = self.input_buffers.input_ids[self._anchor_indices[:num_reqs]]
        scores = self.model.model.candidate_selector(
            candidate_ids,
            unary_logits,
            hidden_states,
            anchor_token_ids,
        )
        self._sample_path(candidate_ids, scores, num_reqs)
        if (
            self.enable_adaptive_verification
            and self.adaptive_confidence_source == "selector"
        ):
            p1 = selector_acceptance_confidences(self._selector_scores[:num_reqs])
            if self._selector_calibrate:
                p1 = apply_selector_calibration(
                    p1, self._selector_cal_knots, self._selector_cal_vals
                )
            self.draft_token_confidence_probs[:num_reqs] = p1
        if self.draft_logits is not None:
            self._cache_draft_logits(candidate_ids, num_sample)
