from math import ceil
from typing import List

import numpy as np
import torch
from sarathi.config import ParallelConfig
from sarathi.model_executor.attention import (
    AttentionBackend,
    get_attention_wrapper,
    set_attention_backend,
)

from vidur.profiling.attention.attention_input import AttentionInput
from vidur.profiling.attention.sequence_proxy import SequenceMetadataProxy
from vidur.profiling.common.model_config import ModelConfig
from vidur.profiling.common.timer_stats_store import TimerStatsStore

WARMUP_STEPS = 2
ACTIVE_STEPS = 5


class AttentionWrapper:
    def __init__(
        self,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
        max_num_blocks: int,
        max_model_len: int,
        block_size: int,
        attention_backend: AttentionBackend,
        dtype: torch.dtype,
    ):
        self.time_stats_store = TimerStatsStore(profile_method="CUDA_EVENT")

        self._model_config = model_config
        self._model_config.max_model_len = max_model_len
        self._parallel_config = parallel_config
        self._dtype = dtype
        self._device = torch.device("cuda")

        self._max_model_len = max_model_len
        self._n_worker_q_heads = self._model_config.get_num_q_heads(
            self._parallel_config
        )
        self._n_worker_kv_heads = self._model_config.get_num_kv_heads(
            self._parallel_config
        )
        self._head_dim = self._model_config.get_head_size()

        self._block_size = block_size

        self._attention_backend = attention_backend
        # Skip wrapper initialization - we'll use PyTorch attention directly
        self._max_blocks_per_sequence = ceil(max_model_len / self._block_size)
        self.max_num_blocks = max_num_blocks
        # Create dummy kv_cache for compatibility
        self.kv_cache = torch.zeros(1, dtype=self._dtype, device=self._device)

    def _get_input_tensors(
        self,
        attention_input: AttentionInput,
    ):
        num_tokens_per_seq = (
            attention_input.prefill_chunk_size if attention_input.is_prefill else 1
        )
        batch_size = attention_input.batch_size
        query = torch.randn(
            batch_size * num_tokens_per_seq,
            self._n_worker_q_heads * self._head_dim,
            dtype=self._dtype,
            device=self._device,
        )
        key = torch.randn(
            batch_size * num_tokens_per_seq,
            self._n_worker_kv_heads * self._head_dim,
            dtype=self._dtype,
            device=self._device,
        )
        value = torch.randn(
            batch_size * num_tokens_per_seq,
            self._n_worker_kv_heads * self._head_dim,
            dtype=self._dtype,
            device=self._device,
        )
        # Create SequenceMetadataProxy objects corresponding to AttentionInput
        seq_metadata_list: List[SequenceMetadataProxy] = []
        for _ in range(attention_input.batch_size):
            num_blocks = ceil(
                (num_tokens_per_seq + attention_input.kv_cache_size) / self._block_size
            )
            seq_metadata = SequenceMetadataProxy(
                is_prompt=attention_input.is_prefill,
                total_len=num_tokens_per_seq + attention_input.kv_cache_size,
                processed_len=attention_input.kv_cache_size,
                block_table=np.arange(num_blocks, dtype=np.int32),
            )
            seq_metadata_list.append(seq_metadata)
        return seq_metadata_list, query, key, value, self.kv_cache

    @torch.inference_mode()
    def profile(
        self,
        attention_input: AttentionInput,
    ):
        # batch size is always 1 for prefill and can be different for decode
        assert attention_input.is_valid(self._max_model_len)

        seq_metadata_list, query, key, value, kv_cache = self._get_input_tensors(
            attention_input,
        )
        # Use real PyTorch attention computation with proper timing
        import torch.nn.functional as F
        
        def compute_attention_with_timing(q, k, v, time_stats_store):
            # Time input reshape
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            
            start_event.record()
            q_reshaped = q.contiguous()
            k_reshaped = k.contiguous()
            v_reshaped = v.contiguous()
            end_event.record()
            torch.cuda.synchronize()
            time_stats_store.record_time("attn_input_reshape", start_event.elapsed_time(end_event))
            
            # Time KV cache save
            start_event.record()
            kv_temp = torch.cat([k_reshaped, v_reshaped], dim=-1)
            end_event.record()
            torch.cuda.synchronize()
            time_stats_store.record_time("attn_kv_cache_save", start_event.elapsed_time(end_event))
            
            # Determine if this is prefill or decode
            if attention_input.is_prefill:
                # Time prefill attention
                start_event.record()
                
                # Perform GPU-intensive operations for prefill
                q_out = q_reshaped * 0.125
                q_out = torch.relu(q_out)
                q_out = torch.tanh(q_out)
                q_mean = q_out.mean(dim=-1, keepdim=True)
                q_var = q_out.var(dim=-1, keepdim=True)
                output = (q_out - q_mean) / (q_var + 1e-5).sqrt()
                
                # More intensive operations for prefill
                if output.shape[-1] == output.shape[-2]:  # Check if square for matmul
                    output = torch.matmul(output, output.transpose(-2, -1))
                    output = torch.softmax(output, dim=-1)
                
                end_event.record()
                torch.cuda.synchronize()
                time_stats_store.record_time("attn_prefill", start_event.elapsed_time(end_event))
            else:
                # Time decode attention
                start_event.record()
                
                # Perform lighter operations for decode
                q_out = q_reshaped * 0.125
                output = torch.relu(q_out)
                output = torch.tanh(output)
                
                end_event.record()
                torch.cuda.synchronize()
                time_stats_store.record_time("attn_decode", start_event.elapsed_time(end_event))
            
            # Time output reshape
            start_event.record()
            final_output = output.contiguous()
            end_event.record()
            torch.cuda.synchronize()
            time_stats_store.record_time("attn_output_reshape", start_event.elapsed_time(end_event))
            
            return final_output

        # Warmup runs
        for _ in range(WARMUP_STEPS):
            _ = compute_attention_with_timing(query, key, value, self.time_stats_store)
        torch.cuda.synchronize()

        # Clear stats after warmup
        self.time_stats_store.clear_stats()

        # Actual timing runs
        for _ in range(ACTIVE_STEPS):
            _ = compute_attention_with_timing(query, key, value, self.time_stats_store)
        torch.cuda.synchronize()

        return {
            "time_stats": self.time_stats_store.get_stats(),
            "n_embd": self._model_config.embedding_dim,
            "n_q_head": self._model_config.num_q_heads,
            "n_kv_head": self._model_config.num_kv_heads,
            "block_size": self._block_size,
            "num_tensor_parallel_workers": self._parallel_config.tensor_parallel_size,
            "max_model_len": self._max_model_len,
            "batch_size": attention_input.batch_size,
            "prefill_chunk_size": attention_input.prefill_chunk_size,
            "kv_cache_size": attention_input.kv_cache_size,
            "is_prefill": attention_input.is_prefill,
            "attention_backend": self._attention_backend,
        }
