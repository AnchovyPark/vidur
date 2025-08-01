from typing import Optional

import torch
try:
    from sarathi.model_executor.layers.activation import SiluAndMul
    from sarathi.model_executor.layers.layernorm import RMSNorm
    from sarathi.model_executor.layers.rotary_embedding import get_rope
    from sarathi.model_executor.parallel_utils.tensor_parallel.layers import (
        ColumnParallelLinear,
        RowParallelLinear,
        VocabParallelEmbedding,
    )
except ImportError:
    # Mock implementations for sarathi layers
    class SiluAndMul(torch.nn.Module):
        def forward(self, x):
            gate, up = x.chunk(2, dim=-1)
            return torch.nn.functional.silu(gate) * up
    
    class RMSNorm(torch.nn.Module):
        def __init__(self, hidden_size, eps=1e-6):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(hidden_size))
            self.variance_epsilon = eps
        
        def forward(self, hidden_states):
            input_dtype = hidden_states.dtype
            hidden_states = hidden_states.to(torch.float32)
            variance = hidden_states.pow(2).mean(-1, keepdim=True)
            hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
            return self.weight * hidden_states.to(input_dtype)
    
    def get_rope(head_size, rotary_dim, max_position, base, is_neox_style=True, rope_scaling=None):
        # Mock implementation
        class MockRoPE:
            def forward(self, positions, query, key):
                return query, key
            
            def __call__(self, positions, query, key):
                return query, key
        return MockRoPE()
    
    class ColumnParallelLinear(torch.nn.Module):
        def __init__(self, input_size, output_size, bias=True, world_size=1, gather_output=True, **kwargs):
            super().__init__()
            # For tensor parallelism, divide output across workers
            self.world_size = world_size
            self.gather_output = gather_output
            self.output_size_per_partition = output_size // world_size
            self.linear = torch.nn.Linear(input_size, self.output_size_per_partition, bias=bias)
        
        def forward(self, x):
            output = self.linear(x)
            # If gather_output is False, return partitioned output
            # If gather_output is True, simulate gathering by returning full size
            if self.gather_output:
                # Simulate gather by expanding the output
                batch_size, seq_len = output.shape[:2]
                full_output = output.repeat(1, 1, self.world_size)
                return full_output, None
            else:
                return output, None
    
    class RowParallelLinear(torch.nn.Module):
        def __init__(self, input_size, output_size, bias=True, world_size=1, **kwargs):
            super().__init__()
            # For tensor parallelism, divide input across workers  
            self.world_size = world_size
            self.input_size_per_partition = input_size // world_size
            self.linear = torch.nn.Linear(self.input_size_per_partition, output_size, bias=bias)
        
        def forward(self, x):
            # In real tensor parallelism, input would be split across devices
            # For simulation, we split the input tensor along the last dimension
            if len(x.shape) == 3:
                partitioned_input = x[:, :, :self.input_size_per_partition]
            else:  # 2D tensor
                partitioned_input = x[:, :self.input_size_per_partition]
            output = self.linear(partitioned_input)
            # In real implementation, outputs would be all-reduced across devices
            return output, None
    
    class VocabParallelEmbedding(torch.nn.Module):
        def __init__(self, num_embeddings, embedding_dim, **kwargs):
            super().__init__()
            self.embedding = torch.nn.Embedding(num_embeddings, embedding_dim)
        
        def forward(self, x):
            return self.embedding(x)

from vidur.profiling.common.cuda_timer import CudaTimer
from vidur.profiling.common.model_config import ModelConfig

REUSE_MEMORY = True


class CausalSelfAttention(torch.nn.Module):

    def __init__(self, config: ModelConfig, world_size: int):
        super().__init__()
        assert config.embedding_dim % config.num_q_heads == 0
        assert config.embedding_dim % world_size == 0
        assert config.num_q_heads % world_size == 0
        assert config.num_kv_heads % world_size == 0

        self.head_dim = config.embedding_dim // config.num_q_heads
        self.num_q_heads_per_worker = config.num_q_heads // world_size
        self.num_kv_heads_per_worker = config.num_kv_heads // world_size

        # For tensor parallelism, each worker handles a portion of heads
        # Since gather_output=False, we need to account for further partitioning
        self.q_size = (self.num_q_heads_per_worker * self.head_dim) // world_size
        self.kv_size = (self.num_kv_heads_per_worker * self.head_dim) // world_size
        self.scaling = self.head_dim**-0.5

        self.qkv_proj = ColumnParallelLinear(
            config.embedding_dim,
            (self.num_q_heads_per_worker + 2 * self.num_kv_heads_per_worker) * self.head_dim,
            bias=config.use_bias or config.use_qkv_bias,
            gather_output=False,
            linear_metric_name="attn_pre_proj",
            world_size=world_size,
        )

        self.o_proj = RowParallelLinear(
            self.num_q_heads_per_worker * self.head_dim,
            config.embedding_dim,
            bias=config.use_bias,
            input_is_parallel=True,
            reduce_results=False,
            linear_metric_name="attn_post_proj",
            world_size=world_size,
        )
        self.rotary_emb = None
        if isinstance(config.rope_theta, int) or isinstance(config.rope_theta, float):
            self.rotary_emb = get_rope(
                self.head_dim,
                rotary_dim=self.head_dim,
                max_position=config.max_model_len,
                base=config.rope_theta,
                is_neox_style=config.is_neox_style,
                rope_scaling=config.rope_scaling,
            )
        self._attn_rope_timer = CudaTimer("attn_rope")

    def forward(self, hidden_states, positions):
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        with self._attn_rope_timer:
            q, k = self.rotary_emb(positions, q, k)
        # output from attn has the same shape as q
        attn_output = torch.randn_like(q)
        output, _ = self.o_proj(attn_output)
        return output


class MLP(torch.nn.Module):
    def __init__(self, config: ModelConfig, world_size: int):
        super().__init__()

        assert config.embedding_dim % world_size == 0

        self.mlp_hidden_dim_per_worker = config.mlp_hidden_dim // world_size
        
        if config.use_gated_mlp:
            self.up_proj = ColumnParallelLinear(
                config.embedding_dim,
                2 * self.mlp_hidden_dim_per_worker,
                bias=config.use_bias,
                gather_output=False,
                world_size=world_size,
                linear_metric_name="mlp_up_proj",
            )
            self.act = SiluAndMul()
        else:
            self.up_proj = ColumnParallelLinear(
                config.embedding_dim,
                self.mlp_hidden_dim_per_worker,
                bias=config.use_bias,
                gather_output=False,
                world_size=world_size,
                linear_metric_name="mlp_up_proj",
            )
            self.act = torch.nn.GELU()

        self.down_proj = RowParallelLinear(
            self.mlp_hidden_dim_per_worker,
            config.embedding_dim,
            bias=config.use_bias,
            input_is_parallel=True,
            world_size=world_size,
            reduce_results=False,
            linear_metric_name="mlp_down_proj",
        )

        self.mlp_act_timer = CudaTimer("mlp_act")

    def forward(self, hidden_states):
        hidden_states, _ = self.up_proj(hidden_states)
        with self.mlp_act_timer:
            hidden_states = self.act(hidden_states)
        hidden_states, _ = self.down_proj(hidden_states)
        return hidden_states


class GPTBlock(torch.nn.Module):

    def __init__(self, config: ModelConfig, world_size: int):
        super().__init__()

        if config.norm == "layer_norm":
            self.input_layernorm = torch.nn.LayerNorm(config.embedding_dim)
        elif config.norm == "rms_norm":
            self.input_layernorm = RMSNorm(config.embedding_dim)
        else:
            raise ValueError(f"Unknown norm: {config.norm} for input_layernorm")

        self._post_attn_norm = config.post_attn_norm
        if config.post_attn_norm:
            if config.norm == "rms_norm":
                self.post_attention_layernorm = RMSNorm(config.embedding_dim)
            else:
                raise ValueError(
                    f"Unknown norm: {config.norm} for post_attention_layernorm"
                )

        self.attn = CausalSelfAttention(config, world_size)
        self.mlp = MLP(config, world_size)

        self.input_layernorm_timer = CudaTimer("input_layernorm")
        self.post_attention_layernorm_timer = CudaTimer("post_attention_layernorm")
        self.add_timer = CudaTimer("add")

    def forward(self, positions, hidden_states, residual):
        if self._post_attn_norm:
            return self._forward_with_post_attn_norm(positions, hidden_states, residual)
        else:
            return self._forward_without_post_attn_norm(positions, hidden_states)

    def _forward_with_post_attn_norm(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
    ):
        # Self Attention
        residual = hidden_states
        with self.input_layernorm_timer:
            hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.attn(
            positions=positions,
            hidden_states=hidden_states,
        )
        hidden_states = residual + hidden_states
        # Fully Connected
        residual = hidden_states
        with self.post_attention_layernorm_timer:
            hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        with self.add_timer:
            hidden_states = residual + hidden_states
        return hidden_states

    def _forward_without_post_attn_norm(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ):
        residual = hidden_states
        with self.input_layernorm_timer:
            hidden_states = self.input_layernorm(hidden_states)
        attn_outputs = self.attn(
            positions=positions,
            hidden_states=hidden_states,
        )
        feed_forward_hidden_states = self.mlp(hidden_states)
        with self.add_timer:
            hidden_states = attn_outputs + feed_forward_hidden_states + residual
        return hidden_states


class GPTModel(torch.nn.Module):
    def __init__(self, config: ModelConfig, world_size: int, num_repeat_steps: int = 1):
        super().__init__()

        self.num_repeat_steps = num_repeat_steps

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.embedding_dim,
            linear_metric_name="emb",
            reduce_results=False,
            world_size=world_size,
            rank=0,
        )

        self.block = GPTBlock(config, world_size=world_size)

    def forward(self, input_ids, positions):
        hidden_states = self.embed_tokens(input_ids)
        residual = hidden_states
        for _ in range(self.num_repeat_steps):
            hidden_states = self.embed_tokens(input_ids)
            hidden_states = self.block(
                positions,
                hidden_states,
                residual,
            )

        return hidden_states
