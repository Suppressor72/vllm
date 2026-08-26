# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for GDNAttentionMetadataBuilder.build() — specifically the
reclassification of non-spec decodes as prefills when spec decodes exist.
Covers the fix for https://github.com/vllm-project/vllm/issues/34845.

Also covers ragged (varlen) spec-decode metadata under adaptive
verification (https://github.com/vllm-project/vllm/issues/51869): the
device query_start_loc carries the trimmed per-request verification
lengths while the CPU one keeps the evenly distributed budget, with
totals conserved.
"""

from dataclasses import dataclass

import pytest
import torch

from tests.v1.attention.utils import (
    BatchSpec,
    create_common_attn_metadata,
    create_vllm_config,
)
from vllm.config import SpeculativeConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.attention.backends.gdn_attn import (
    GDNAttentionBackend,
    GDNAttentionMetadata,
    GDNAttentionMetadataBuilder,
)
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID
from vllm.v1.kv_cache_interface import MambaSpec

BLOCK_SIZE = 16
DEVICE = torch.device("cpu")


@dataclass
class GDNBuildTestCase:
    """Specification for a GDN metadata builder classification test."""

    seq_lens: list[int]
    query_lens: list[int]
    num_decode_draft_tokens: list[int] | None  # None = no spec config
    num_speculative_tokens: int
    expected_num_decodes: int
    expected_num_prefills: int
    expected_num_prefill_tokens: int
    expected_num_spec_decodes: int


GDN_BUILD_TEST_CASES = {
    # The original #34845 crash: non-spec query_len=1 + spec decode
    "mixed_decode_and_spec_decode": GDNBuildTestCase(
        seq_lens=[65, 20],
        query_lens=[1, 3],
        num_decode_draft_tokens=[-1, 2],
        num_speculative_tokens=2,
        expected_num_decodes=0,
        expected_num_prefills=1,
        expected_num_prefill_tokens=1,
        expected_num_spec_decodes=1,
    ),
    # All requests are spec decodes — no reclassification needed
    "pure_spec_decode": GDNBuildTestCase(
        seq_lens=[50, 30],
        query_lens=[3, 3],
        num_decode_draft_tokens=[2, 2],
        num_speculative_tokens=2,
        expected_num_decodes=0,
        expected_num_prefills=0,
        expected_num_prefill_tokens=0,
        expected_num_spec_decodes=2,
    ),
    # No speculative config at all — standard decode path
    "pure_regular_decode": GDNBuildTestCase(
        seq_lens=[40, 30, 20],
        query_lens=[1, 1, 1],
        num_decode_draft_tokens=None,
        num_speculative_tokens=0,
        expected_num_decodes=3,
        expected_num_prefills=0,
        expected_num_prefill_tokens=0,
        expected_num_spec_decodes=0,
    ),
    # Multi-token prefill alongside spec decode — no decode to reclassify
    "spec_decode_with_real_prefill": GDNBuildTestCase(
        seq_lens=[100, 20],
        query_lens=[50, 3],
        num_decode_draft_tokens=[-1, 2],
        num_speculative_tokens=2,
        expected_num_decodes=0,
        expected_num_prefills=1,
        expected_num_prefill_tokens=50,
        expected_num_spec_decodes=1,
    ),
    # All three types in one batch — decode gets reclassified
    "prefill_decode_and_spec_decode": GDNBuildTestCase(
        seq_lens=[100, 65, 20],
        query_lens=[50, 1, 3],
        num_decode_draft_tokens=[-1, -1, 2],
        num_speculative_tokens=2,
        expected_num_decodes=0,
        expected_num_prefills=2,
        expected_num_prefill_tokens=51,
        expected_num_spec_decodes=1,
    ),
    # Multiple non-spec query_len=1 requests all reclassified
    "multiple_decodes_reclassified": GDNBuildTestCase(
        seq_lens=[40, 50, 60, 20],
        query_lens=[1, 1, 1, 3],
        num_decode_draft_tokens=[-1, -1, -1, 2],
        num_speculative_tokens=2,
        expected_num_decodes=0,
        expected_num_prefills=3,
        expected_num_prefill_tokens=3,
        expected_num_spec_decodes=1,
    ),
    # Zero-length padded sequence excluded from counts
    "zero_length_padding_with_spec": GDNBuildTestCase(
        seq_lens=[16, 65, 20],
        query_lens=[0, 1, 3],
        num_decode_draft_tokens=[-1, -1, 2],
        num_speculative_tokens=2,
        expected_num_decodes=0,
        expected_num_prefills=1,
        expected_num_prefill_tokens=1,
        expected_num_spec_decodes=1,
    ),
}


def _create_gdn_builder(
    num_speculative_tokens: int = 0,
    full_cuda_graph: bool = False,
    realistic_spec_blocks: bool = False,
) -> GDNAttentionMetadataBuilder:
    """Create a GDNAttentionMetadataBuilder with minimal config."""
    vllm_config = create_vllm_config(
        model_name="Qwen/Qwen3.5-0.8B",
        block_size=BLOCK_SIZE,
    )
    # Pin the mode so the full-cudagraph buffer branch only runs when asked:
    # the platform default resolves to a full-capable mode on CUDA hosts.
    vllm_config.compilation_config.cudagraph_mode = (
        CUDAGraphMode.FULL_AND_PIECEWISE if full_cuda_graph else CUDAGraphMode.PIECEWISE
    )
    if num_speculative_tokens > 0:
        vllm_config.speculative_config = SpeculativeConfig(
            method="ngram",
            num_speculative_tokens=num_speculative_tokens,
        )
    mamba_spec = MambaSpec(
        block_size=BLOCK_SIZE,
        shapes=((16, 64),),
        dtypes=(torch.float16,),
        # Production derives num_speculative_blocks from
        # num_speculative_tokens (see MambaMixerMixin); the ragged tests
        # use the production-shaped spec so align-mode gathering reads a
        # distinct (num_spec + 1)-wide state row per request.
        num_speculative_blocks=(num_speculative_tokens if realistic_spec_blocks else 0),
    )
    return GDNAttentionMetadataBuilder(
        kv_cache_spec=mamba_spec,
        layer_names=["layer.0"],
        vllm_config=vllm_config,
        device=DEVICE,
    )


def _build(
    builder: GDNAttentionMetadataBuilder,
    batch_spec: BatchSpec,
    num_decode_draft_tokens: list[int] | None = None,
) -> GDNAttentionMetadata:
    """Build GDN attention metadata, optionally with spec-decode kwargs."""
    common = create_common_attn_metadata(batch_spec, BLOCK_SIZE, DEVICE)
    kwargs: dict = {}
    if num_decode_draft_tokens is not None:
        kwargs["num_decode_draft_tokens_cpu"] = torch.tensor(
            num_decode_draft_tokens, dtype=torch.int32
        )
        kwargs["num_accepted_tokens"] = torch.ones(
            batch_spec.batch_size, dtype=torch.int32, device=DEVICE
        )
    return builder.build(common_prefix_len=0, common_attn_metadata=common, **kwargs)


@pytest.mark.parametrize(
    "test_case", GDN_BUILD_TEST_CASES.values(), ids=GDN_BUILD_TEST_CASES.keys()
)
def test_gdn_build_classification(test_case: GDNBuildTestCase):
    """Test that GDN metadata builder classifies requests correctly."""
    builder = _create_gdn_builder(test_case.num_speculative_tokens)
    batch = BatchSpec(seq_lens=test_case.seq_lens, query_lens=test_case.query_lens)
    meta = _build(builder, batch, test_case.num_decode_draft_tokens)

    assert meta.num_decodes == test_case.expected_num_decodes
    assert meta.num_prefills == test_case.expected_num_prefills
    assert meta.num_prefill_tokens == test_case.expected_num_prefill_tokens
    assert meta.num_spec_decodes == test_case.expected_num_spec_decodes


def test_has_initial_state_after_reclassification():
    """After reclassification, num_prefills > 0 so the prefill kernel path
    should compute has_initial_state. For the reclassified request with
    context_lens > 0, the corresponding entry must be True."""
    builder = _create_gdn_builder(num_speculative_tokens=2)
    batch = BatchSpec(seq_lens=[65, 20], query_lens=[1, 3])
    meta = _build(builder, batch, num_decode_draft_tokens=[-1, 2])

    assert meta.num_prefills > 0, "reclassification should produce prefills"
    assert meta.has_initial_state is not None
    # req0 has context_lens = 65 - 1 = 64 > 0, so has_initial_state[0] = True
    assert meta.has_initial_state[0].item() is True


def test_full_cudagraph_spec_metadata_uses_request_count():
    """FULL cudagraph token padding must not pad request-indexed metadata."""
    num_speculative_tokens = 3
    builder = _create_gdn_builder(
        num_speculative_tokens=num_speculative_tokens,
        full_cuda_graph=True,
    )
    batch = BatchSpec(seq_lens=[80, 96], query_lens=[4, 4])
    meta = _build(builder, batch, num_decode_draft_tokens=[3, 3])

    assert meta.num_spec_decodes == batch.batch_size
    assert meta.num_spec_decode_tokens == batch.compute_num_tokens()
    assert meta.spec_state_indices_tensor is not None
    assert meta.spec_state_indices_tensor.shape == (
        batch.batch_size,
        num_speculative_tokens + 1,
    )
    assert meta.spec_sequence_masks is not None
    assert meta.spec_sequence_masks.shape == (batch.batch_size,)
    assert meta.spec_query_start_loc is not None
    assert meta.spec_query_start_loc.shape == (batch.batch_size + 1,)
    assert meta.num_accepted_tokens is not None
    assert meta.num_accepted_tokens.shape == (batch.batch_size,)


def _build_ragged(
    builder: GDNAttentionMetadataBuilder,
    batch_spec: BatchSpec,
    device_query_lens: list[int],
    num_decode_draft_tokens: list[int],
    num_accepted_tokens: list[int],
) -> tuple[GDNAttentionMetadata, object]:
    """Build metadata where the device cu_seqlens diverge from the CPU ones.

    Mimics adaptive verification: the CPU keeps the evenly distributed
    draft budget (what ``batch_spec.query_lens`` encodes), while the device
    query_start_loc carries the device-side reallocation. Totals conserved.
    Returns the metadata and the common metadata it was built from.
    """
    common = create_common_attn_metadata(batch_spec, BLOCK_SIZE, DEVICE)
    # Production-shaped block table: align-mode gathering indexes
    # start_indices + [0..num_speculative_blocks], and production reserves
    # exactly that speculative-block width beyond cdiv(max_seq_len,
    # block_size) (MambaSpec.max_num_blocks_per_req).
    max_blocks = (max(batch_spec.seq_lens) + BLOCK_SIZE - 1) // BLOCK_SIZE
    width = max_blocks + builder.num_spec
    common.block_table_tensor = torch.arange(
        batch_spec.batch_size * width, dtype=torch.int32, device=DEVICE
    ).view(batch_spec.batch_size, width)
    assert sum(device_query_lens) == batch_spec.compute_num_tokens()
    # Load-bearing scheduler invariant: no trimmed verification window may
    # exceed the spec state-index row width (num_spec + 1).
    spec_lens = [
        ql for ql, d in zip(device_query_lens, num_decode_draft_tokens) if d >= 0
    ]
    assert max(spec_lens, default=0) <= builder.num_spec + 1
    # Contract: the rejection sampler always reports at least the bonus
    # token (num_accepted_tokens >= 1); the Triton kernels compute
    # num_accepted - 1 without a zero guard.
    assert min(num_accepted_tokens) >= 1
    trimmed = torch.zeros(
        common.query_start_loc.size(0), dtype=torch.int32, device=DEVICE
    )
    trimmed[1:] = torch.tensor(device_query_lens, dtype=torch.int32).cumsum(0)
    common.query_start_loc = trimmed
    meta = builder.build(
        common_prefix_len=0,
        common_attn_metadata=common,
        num_accepted_tokens=torch.tensor(
            num_accepted_tokens, dtype=torch.int32, device=DEVICE
        ),
        num_decode_draft_tokens_cpu=torch.tensor(
            num_decode_draft_tokens, dtype=torch.int32
        ),
    )
    return meta, common


def test_ragged_spec_decode_builds_device_truth_metadata():
    """Pure spec batch: per-request spec boundaries must come from the
    device query_start_loc, not the CPU even distribution."""
    # K=2; CPU even budget = 2 query tokens per request ([1,1,1] drafts),
    # device reallocates to [3,2,1] — both total 6.
    builder = _create_gdn_builder(num_speculative_tokens=2, realistic_spec_blocks=True)
    batch = BatchSpec(seq_lens=[80, 80, 80], query_lens=[2, 2, 2])
    meta, common = _build_ragged(
        builder,
        batch,
        device_query_lens=[3, 2, 1],
        num_decode_draft_tokens=[1, 1, 1],
        num_accepted_tokens=[2, 1, 1],
    )

    assert meta.num_spec_decodes == 3
    assert meta.num_spec_decode_tokens == 6
    # Device truth, not the CPU even [0,2,4,6].
    assert meta.spec_query_start_loc.tolist() == [0, 3, 5, 6]
    assert meta.spec_token_indx.tolist() == [0, 1, 2, 3, 4, 5]
    assert meta.spec_sequence_masks.tolist() == [True, True, True]
    # Explicit expected state rows: the arange table has width
    # max_blocks(5) + num_spec(2) = 7, so request i's row starts at 7*i,
    # and align mode gathers (seq_len-1)//16 + [0,1,2] = [4,5,6].
    assert meta.spec_state_indices_tensor.tolist() == [
        [4, 5, 6],
        [11, 12, 13],
        [18, 19, 20],
    ]
    assert meta.num_accepted_tokens.tolist() == [2, 1, 1]


def test_ragged_spec_lens_in_mixed_batch():
    """Prefill + ragged spec decodes: the token partition must follow the
    device lengths while prefill classification stays CPU-exact."""
    builder = _create_gdn_builder(num_speculative_tokens=2, realistic_spec_blocks=True)
    batch = BatchSpec(seq_lens=[200, 80, 80, 80], query_lens=[50, 2, 2, 2])
    meta, _ = _build_ragged(
        builder,
        batch,
        device_query_lens=[50, 3, 2, 1],
        num_decode_draft_tokens=[-1, 1, 1, 1],
        num_accepted_tokens=[1, 2, 1, 1],
    )

    assert meta.num_prefills == 1
    assert meta.num_prefill_tokens == 50
    assert meta.num_spec_decodes == 3
    assert meta.num_spec_decode_tokens == 6
    assert meta.non_spec_query_start_loc.tolist() == [0, 50]
    assert meta.spec_query_start_loc.tolist() == [0, 3, 5, 6]
    # Stable partition: non-spec tokens first, spec tokens last.
    assert meta.non_spec_token_indx.tolist() == list(range(50))
    assert meta.spec_token_indx.tolist() == list(range(50, 56))
    # Explicit expected state rows: table width max_blocks(13) + 2 = 15;
    # spec requests are indices 1..3 with start (80-1)//16 = 4, so their
    # rows start at 15, 30, 45 and gather [4,5,6].
    assert meta.spec_state_indices_tensor.tolist() == [
        [19, 20, 21],
        [34, 35, 36],
        [49, 50, 51],
    ]


def test_ragged_spec_lens_full_cudagraph_padding():
    """FULL-cudagraph replay with ragged lens: the fourth (padded) request
    slot must hold the pad sentinels while the real rows carry device
    truth, all inside the builder's persistent buffers."""
    builder = _create_gdn_builder(
        num_speculative_tokens=2,
        full_cuda_graph=True,
        realistic_spec_blocks=True,
    )
    batch = BatchSpec(seq_lens=[80, 80, 80, 16], query_lens=[2, 2, 2, 0])
    meta, _ = _build_ragged(
        builder,
        batch,
        device_query_lens=[3, 2, 1, 0],
        num_decode_draft_tokens=[1, 1, 1, -1],
        num_accepted_tokens=[2, 1, 1, 1],
    )

    # Metadata is sliced to the padded request count and lives in the
    # builder's persistent buffers (what a replayed graph reads).
    assert (
        meta.spec_query_start_loc.data_ptr() == builder.spec_query_start_loc.data_ptr()
    )
    assert meta.spec_query_start_loc.tolist() == [0, 3, 5, 6, 6]
    assert meta.spec_sequence_masks.tolist() == [True, True, True, False]
    assert meta.num_accepted_tokens.tolist() == [2, 1, 1, 1]
    # Explicit expected rows (table width 5 + 2 = 7, gather [4,5,6] per
    # real request) with the padded fourth slot nulled out.
    assert meta.spec_state_indices_tensor.tolist() == [
        [4, 5, 6],
        [11, 12, 13],
        [18, 19, 20],
        [NULL_BLOCK_ID] * (builder.num_spec + 1),
    ]


def test_gdn_backend_supports_device_cpu_query_lens_mismatch():
    """GDN overrides the SSM blanket opt-out; mamba must stay gated."""
    from vllm.v1.attention.backend import AttentionBackend
    from vllm.v1.attention.backends.mamba1_attn import Mamba1AttentionBackend

    assert GDNAttentionBackend.is_ssm()
    assert GDNAttentionBackend.supports_device_cpu_query_lens_mismatch()
    assert Mamba1AttentionBackend.is_ssm()
    assert not Mamba1AttentionBackend.supports_device_cpu_query_lens_mismatch()
    # The base default still opts SSM-family backends out.
    assert AttentionBackend.supports_device_cpu_query_lens_mismatch() == (
        not AttentionBackend.is_ssm()
    )
