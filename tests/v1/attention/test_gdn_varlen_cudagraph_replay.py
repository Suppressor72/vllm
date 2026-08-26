# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Real CUDA-graph capture/replay test for GDN varlen (ragged) decode
metadata — the mechanism adaptive verification's varlen decode cudagraphs
depend on (https://github.com/vllm-project/vllm/issues/51869).

A graph is captured while the builder's persistent buffers hold a uniform
spec batch; the buffers are then re-filled by a ragged rebuild (device
[3,2,1] vs CPU-even [2,2,2], totals conserved), and the SAME graph is
replayed without re-capture. The replay must observe the re-filled,
device-truth values, including the padded fourth request slot.
"""

import pytest
import torch

from tests.v1.attention.utils import (
    BatchSpec,
    create_common_attn_metadata,
    create_vllm_config,
)
from vllm.config import SpeculativeConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadataBuilder
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID
from vllm.v1.kv_cache_interface import MambaSpec

BLOCK_SIZE = 16
DEVICE = torch.device("cuda")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA for graph capture"
)


def _create_builder() -> GDNAttentionMetadataBuilder:
    vllm_config = create_vllm_config(
        model_name="Qwen/Qwen3.5-0.8B", block_size=BLOCK_SIZE
    )
    vllm_config.compilation_config.cudagraph_mode = CUDAGraphMode.FULL_AND_PIECEWISE
    vllm_config.speculative_config = SpeculativeConfig(
        method="ngram",
        num_speculative_tokens=2,
    )
    mamba_spec = MambaSpec(
        block_size=BLOCK_SIZE,
        shapes=((16, 64),),
        dtypes=(torch.float16,),
        # Production-shaped spec: align-mode gathering reads a distinct
        # (num_spec + 1)-wide state row per request.
        num_speculative_blocks=2,
    )
    return GDNAttentionMetadataBuilder(
        kv_cache_spec=mamba_spec,
        layer_names=["layer.0"],
        vllm_config=vllm_config,
        device=DEVICE,
    )


def _build_into_buffers(
    builder: GDNAttentionMetadataBuilder,
    device_query_lens: list[int],
    num_accepted_tokens: list[int],
):
    """Build into the persistent buffers: 3 real spec requests with the
    CPU-even draft budget ([1,1,1] drafts → query lens [2,2,2]) plus one
    padded non-spec slot; the device cu_seqlens may be trimmed/realloacted
    (adaptive verification), totals conserved at 6."""
    batch = BatchSpec(seq_lens=[80, 80, 80, 16], query_lens=[2, 2, 2, 0])
    common = create_common_attn_metadata(
        batch, BLOCK_SIZE, DEVICE, arange_block_indices=True
    )
    # Production-shaped block table: align-mode gathering indexes
    # start_indices + [0..num_speculative_blocks], and production reserves
    # exactly that speculative-block width beyond cdiv(max_seq_len,
    # block_size) (MambaSpec.max_num_blocks_per_req) — width 5 + 2 = 7.
    width = (max(batch.seq_lens) + BLOCK_SIZE - 1) // BLOCK_SIZE + builder.num_spec
    common.block_table_tensor = torch.arange(
        batch.batch_size * width, dtype=torch.int32, device=DEVICE
    ).view(batch.batch_size, width)
    assert sum(device_query_lens) == batch.compute_num_tokens()
    # All four slots are verification/padded slots in this fixture; the
    # rejection sampler always reports >= 1 accepted token (the Triton
    # kernels compute num_accepted - 1 without a zero guard).
    assert max(device_query_lens) <= builder.num_spec + 1
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
        num_decode_draft_tokens_cpu=torch.tensor([1, 1, 1, -1], dtype=torch.int32),
    )
    return meta, common


def test_ragged_rebuild_observed_by_replayed_cudagraph():
    builder = _create_builder()
    num_spec = builder.num_spec
    batch_size = 4  # 3 real spec requests + 1 padded slot

    # Capture-shape build: uniform device == CPU-even [2,2,2,0].
    meta0, _ = _build_into_buffers(builder, [2, 2, 2, 0], [2, 2, 1, 1])
    assert meta0.spec_query_start_loc.data_ptr() == (
        builder.spec_query_start_loc.data_ptr()
    )

    # The consumer reads ONLY the builder's persistent buffers — the graph
    # bakes their addresses, exactly like a varlen decode FULL graph.
    qsl = builder.spec_query_start_loc[: batch_size + 1]
    states = builder.spec_state_indices_tensor[:batch_size].flatten()
    accepted = builder.num_accepted_tokens[:batch_size]
    masks = builder.spec_sequence_masks[:batch_size].to(torch.int32)
    out = torch.empty(
        qsl.numel() + states.numel() + accepted.numel() + masks.numel(),
        dtype=torch.int32,
        device=DEVICE,
    )

    def consumer():
        out.copy_(torch.cat([qsl, states, accepted, masks]))

    # Warm up on a side stream, then capture.
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        consumer()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        consumer()
    capture_snapshot = out.clone()

    # Ragged rebuild: the device reallocates to [3,2,1] while the CPU keeps
    # the even distribution — adaptive verification's divergence.
    meta1, common1 = _build_into_buffers(builder, [3, 2, 1, 0], [2, 1, 1, 1])

    # The eager metadata itself is device-truth...
    assert meta1.spec_query_start_loc.tolist() == [0, 3, 5, 6, 6]
    # ...and the SAME captured graph, replayed without re-capture, must
    # observe the re-filled buffers.
    out.zero_()
    g.replay()
    torch.cuda.synchronize()
    replayed = out.tolist()
    n_qsl = batch_size + 1
    n_states = batch_size * (num_spec + 1)
    assert replayed[:n_qsl] == [0, 3, 5, 6, 6]
    # Explicit expected state rows: request i's row of the width-7 arange
    # table starts at 7*i and align mode gathers (seq_len-1)//16 + [0,1,2]
    # = [4,5,6]; the padded fourth slot is nulled out. The replayed graph
    # must observe exactly these rows.
    assert replayed[n_qsl : n_qsl + n_states] == [
        4,
        5,
        6,
        11,
        12,
        13,
        18,
        19,
        20,
        NULL_BLOCK_ID,
        NULL_BLOCK_ID,
        NULL_BLOCK_ID,
    ]
    assert replayed[n_qsl + n_states : n_qsl + n_states + batch_size] == [2, 1, 1, 1]
    assert replayed[n_qsl + n_states + batch_size :] == [1, 1, 1, 0]
    assert replayed != capture_snapshot.tolist()
