# coding=utf-8
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" PyTorch LLaMA model."""
import copy
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "5"
import math
from typing import List, Optional, Tuple, Union
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn

from transformers.activations import ACT2FN
from huggingface_hub import hf_hub_download


try:
    from .configs import EConfig
    from .utils_c import *
    from .choices import *
except:
    from configs import EConfig
    from utils_c import *
    from choices import *
    from utils import prepare_logits_processor




# Copied from transformers.models.bart.modeling_bart._make_causal_mask
def _make_causal_mask(
        input_ids_shape: torch.Size, dtype: torch.dtype, device: torch.device, past_key_values_length: int = 0
):
    """
    Make causal mask used for bi-directional self-attention.
    """
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask], dim=-1)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)


# Copied from transformers.models.bart.modeling_bart._expand_mask
def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    # The first two dimensions of cos and sin are always 1, so we can `squeeze` them.
    cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
    sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]
    cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class LlamaRotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype()
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :].to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :].to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
            self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
        )


class LlamaLinearScalingRotaryEmbedding(LlamaRotaryEmbedding):
    """LlamaRotaryEmbedding extended with linear scaling. Credits to the Reddit user /u/kaiokendev"""

    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0):
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)
        t = t / self.scaling_factor

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :].to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :].to(dtype), persistent=False)


class LlamaDynamicNTKScalingRotaryEmbedding(LlamaRotaryEmbedding):
    """LlamaRotaryEmbedding extended with Dynamic NTK scaling. Credits to the Reddit users /u/bloc97 and /u/emozilla"""

    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0):
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len

        if seq_len > self.max_position_embeddings:
            base = self.base * (
                    (self.scaling_factor * seq_len / self.max_position_embeddings) - (self.scaling_factor - 1)
            ) ** (self.dim / (self.dim - 2))
            inv_freq = 1.0 / (base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
            self.register_buffer("inv_freq", inv_freq, persistent=False)

        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :].to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :].to(dtype), persistent=False)


class LlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.q_proj = nn.Linear(self.hidden_size * 2, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size * 2, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size * 2, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        self._init_rope()

    def _init_rope(self):
        if self.config.rope_scaling is None:
            if hasattr(self.config, "rope_theta"):
                self.rotary_emb = LlamaRotaryEmbedding(self.head_dim,
                                                       max_position_embeddings=self.max_position_embeddings,
                                                       base=self.config.rope_theta)
            else:
                self.rotary_emb = LlamaRotaryEmbedding(self.head_dim,
                                                       max_position_embeddings=self.max_position_embeddings)
        else:
            scaling_type = self.config.rope_scaling["type"]
            scaling_factor = self.config.rope_scaling["factor"]
            if scaling_type == "linear":
                self.rotary_emb = LlamaLinearScalingRotaryEmbedding(
                    self.head_dim, max_position_embeddings=self.max_position_embeddings, scaling_factor=scaling_factor
                )
            elif scaling_type == "dynamic":
                self.rotary_emb = LlamaDynamicNTKScalingRotaryEmbedding(
                    self.head_dim, max_position_embeddings=self.max_position_embeddings, scaling_factor=scaling_factor
                )
            else:
                raise ValueError(f"Unknown RoPE scaling type {scaling_type}")

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Tuple[torch.Tensor]] = None,
            output_attentions: bool = False,
            use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        if self.config.pretraining_tp > 1:
            key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.config.pretraining_tp
            query_slices = self.q_proj.weight.split(
                (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
            )
            key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
            value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

            query_states = [F.linear(hidden_states, query_slices[i]) for i in range(self.config.pretraining_tp)]
            query_states = torch.cat(query_states, dim=-1)

            key_states = [F.linear(hidden_states, key_slices[i]) for i in range(self.config.pretraining_tp)]
            key_states = torch.cat(key_states, dim=-1)

            value_states = [F.linear(hidden_states, value_slices[i]) for i in range(self.config.pretraining_tp)]
            value_states = torch.cat(value_states, dim=-1)

        else:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        if self.config.pretraining_tp > 1:
            attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
            o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
            attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
        else:
            attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class LlamaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        if self.config.pretraining_tp > 1:
            slice = self.intermediate_size // self.config.pretraining_tp
            gate_proj_slices = self.gate_proj.weight.split(slice, dim=0)
            up_proj_slices = self.up_proj.weight.split(slice, dim=0)
            down_proj_slices = self.down_proj.weight.split(slice, dim=1)

            gate_proj = torch.cat(
                [F.linear(x, gate_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1
            )
            up_proj = torch.cat([F.linear(x, up_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1)

            intermediate_states = (self.act_fn(gate_proj) * up_proj).split(slice, dim=2)
            down_proj = [
                F.linear(intermediate_states[i], down_proj_slices[i]) for i in range(self.config.pretraining_tp)
            ]
            down_proj = sum(down_proj)
        else:
            down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

        return down_proj


class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class LlamaDecoderLayeremb(nn.Module):
    def __init__(self, config, last=True):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = LlamaAttention(config=config)
        self.mlp = LlamaMLP(config)
        self.last = last
        # self.fc = nn.Linear(config.hidden_size * 2, config.hidden_size)
        self.hidden_norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # if self.index!=0:

        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
            self,
            input_emb: torch.Tensor,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Tuple[torch.Tensor]] = None,
            output_attentions: Optional[bool] = False,
            use_cache: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """

        residual = hidden_states

        hidden_states = self.hidden_norm(hidden_states)
        input_emb = self.input_layernorm(input_emb)

        hidden_states = torch.cat((input_emb, hidden_states), dim=-1)


        # cache_hidden.append(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


@torch.no_grad()
def padding(tensor, left=True):
    zeropadding = torch.zeros_like(tensor[:, -1:])
    if left:
        tensor = torch.cat((zeropadding, tensor[:, :-1]), dim=1)
    else:
        tensor = torch.cat((tensor[:, 1:], zeropadding), dim=1)
    return tensor



def len_list(x, n):
    return [i for i in x if len(i) <= n]


class Model(nn.Module):
    def __init__(self, config, load_emb=False, path=None, bias=True, total_tokens=63, depth=5, top_k=8, threshold=1.0,
                 # DDD parameters
                 use_ddd=False, ddd_max_depth=9, ddd_check_steps=None, ddd_threshold=-10.0,
                 use_ddd_dynamic_budget=False, ddd_min_budget=32,
                 # OPT-Tree parameters
                 use_opt_tree=False, opt_expand_factor=2.0):
        super().__init__()
        self.config=config
        self.gradient_checkpointing = True
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.lm_head=nn.Linear(config.hidden_size,config.draft_vocab_size,bias=False)
        if load_emb and not hasattr(config, "target_hidden_size"):
            from safetensors import safe_open
            import json
            try:
                index_json_path = os.path.join(path, "model.safetensors.index.json")
                if not os.path.exists(index_json_path):
                    index_json_path = hf_hub_download(path, "model.safetensors.index.json")
                with open(index_json_path, "r") as f:
                    index_json = json.loads(f.read())
                    emb_path = index_json["weight_map"]["model.embed_tokens.weight"]
                local_emb_path = os.path.join(path, emb_path)
                if not os.path.exists(local_emb_path):
                    local_emb_path = hf_hub_download(path, emb_path)
                with safe_open(local_emb_path,
                               framework="pt",
                               device="cpu") as f:
                    tensor_slice = f.get_slice("model.embed_tokens.weight")
                    vocab_size, hidden_dim = tensor_slice.get_shape()
                    tensor = tensor_slice[:, :hidden_dim].float()
            except:
                index_json_path = os.path.join(path, "pytorch_model.bin.index.json")
                if not os.path.exists(index_json_path):
                    index_json_path = hf_hub_download(path, "pytorch_model.bin.index.json")
                with open(index_json_path, "r") as f:
                    index_json = json.loads(f.read())
                    emb_path = index_json["weight_map"]["model.embed_tokens.weight"]
                local_emb_path = os.path.join(path, emb_path)
                if not os.path.exists(local_emb_path):
                    local_emb_path = hf_hub_download(path, emb_path)
                weights = torch.load(local_emb_path)
                tensor = weights["model.embed_tokens.weight"].float()
            self.embed_tokens.weight.data = tensor

        self.top_k = top_k
        self.total_tokens = total_tokens - 1
        self.depth = depth
        self.threshold = math.log(threshold)

        # ---- DDD (Dynamic Depth Decoding) ----
        self.use_ddd = use_ddd
        self.ddd_max_depth = ddd_max_depth if ddd_max_depth is not None else depth
        if ddd_check_steps is None:
            # Default: check at steps 5, 7, 9 (but capped by max_depth)
            self.ddd_check_steps = [s for s in [5, 7, 9] if s < self.ddd_max_depth]
        else:
            self.ddd_check_steps = ddd_check_steps
        self.ddd_threshold = ddd_threshold
        self.use_ddd_dynamic_budget = use_ddd_dynamic_budget
        self.ddd_min_budget = ddd_min_budget
        # Stats tracking. Keep legacy keys while adding per-call records for
        # threshold profiling.
        self._ddd_stats = {
            "calls": 0,
            "early_stops": 0,
            "total_checks": 0,
            "depths": [],
            "actual_depths": [],
            "early_stop_steps": [],
            "checked_H": [],
            "budget_tokens": [],
            "call_records": [],
        }
        # ---- end DDD ----

        # ---- OPT-Tree ----
        self.use_opt_tree = use_opt_tree
        self.opt_expand_factor = opt_expand_factor
        # ---- end OPT-Tree ----
        # print("total_tokens",total_tokens)
        # print("depth",depth)
        # print("top_k",top_k)
        # print("threshold",threshold)
        self.hidden_size = config.hidden_size
        self.midlayer = LlamaDecoderLayeremb(config)
        if hasattr(config, "target_hidden_size"):
            self.fc = nn.Linear(config.target_hidden_size * 3, self.hidden_size, bias=False)
        else:
            self.fc = nn.Linear(config.hidden_size * 3, self.hidden_size, bias=False)
        self.norm=LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.logsoftmax = nn.LogSoftmax(dim=-1)

        d2t=torch.zeros((config.draft_vocab_size),dtype=torch.long)
        t2d=torch.zeros((config.vocab_size),dtype=torch.bool)
        self.register_buffer("d2t", d2t)
        self.register_buffer("t2d", t2d)

        for param in self.embed_tokens.parameters():
            param.requires_grad = False

    def init_tree(self):
        self.tree_mask_init = torch.eye(self.top_k, device=self.embed_tokens.weight.device)[None, None]
        self.position_ids = torch.zeros(self.top_k, device=self.embed_tokens.weight.device, dtype=torch.long)
        self.tree_mask_init = self.tree_mask_init.to(self.embed_tokens.weight.device)

    def reset(self):
        self.tree_mask = None

    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape,
                # inputs_embeds.dtype,
                torch.float32,  # [MODIFIED] force to cast to float32
                device=inputs_embeds.device,
                past_key_values_length=past_key_values_length,
            )

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = _expand_mask(attention_mask, torch.float32, tgt_len=input_shape[-1]).to(
                inputs_embeds.device
            )
            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
            )

        # [MODIFIED] add tree mask
        if hasattr(self, "tree_mask") and self.tree_mask is not None:
            tree_mask = self.tree_mask
            _, _, tree_shape0, tree_shape1 = tree_mask.shape
            combined_attention_mask[:, :, -tree_shape0:, -tree_shape1:][
                tree_mask == 0
                ] = torch.finfo(torch.float32).min

        return combined_attention_mask

    def forward(
            self,
            hidden_states,
            input_ids,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            std=None
    ):
        batch_size, seq_length, _ = hidden_states.shape
        seq_length_with_past = seq_length
        past_key_values_length = 0

        with torch.no_grad():
            inputs_embeds = self.embed_tokens(input_ids)
            # inputs_embeds = inputs_embeds.detach()

        # if std is not None:
        #     noise = torch.randn(inputs_embeds.size(),device=inputs_embeds.device) * std
        #     inputs_embeds=inputs_embeds+noise

        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length
        if position_ids is None:
            device = hidden_states.device if hidden_states is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()

        #position_ids=position_ids//4
        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length_with_past), dtype=torch.bool, device=hidden_states.device
            )
        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, (batch_size, seq_length), hidden_states, past_key_values_length
        )

        # if self.gradient_checkpointing and self.training:
        #    if use_cache:
        #        use_cache = False

        # hidden_states=self.act(self.fc(torch.cat((inputs_embeds,hidden_states),dim=-1)))
        inputs_embeds = inputs_embeds.to(hidden_states.dtype)
        if hidden_states.shape[-1]!=inputs_embeds.shape[-1]:
            hidden_states = self.fc(hidden_states)
        # hidden_states = self.fc(hidden_states)

        all_hidden_states = () if output_hidden_states else None
        next_decoder_cache = () if use_cache else None

        past_key_value = past_key_values[0] if past_key_values is not None else None
        layer_outputs = self.midlayer(
            input_emb=inputs_embeds,
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=True,
        )
        if use_cache:
            next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)
        hidden_states = layer_outputs[0]


        if use_cache:
            return hidden_states, next_decoder_cache

        return hidden_states

    def reset_kv(self):
        self.stable_kv = None

    def _opt_tree_select(self, scores_flat, token_flat, parents_flat, budget, sample_token):
        """
        OPT-Tree greedy node selection with connectivity guarantee.

        Given an over-expanded set of candidate nodes, select `budget` nodes
        (excluding root) to maximize expected acceptance length:
            E[acc_len] = sum_v exp(score[v])

        Greedy algorithm:
          1. Sort all candidates by exp(score) descending
          2. For each candidate (highest first):
             If adding it (+ all unselected ancestors on path to root)
             doesn't exceed budget, select it + ancestors.

        Args:
            scores_flat:  [M] cumulative log-prob scores for each candidate
            token_flat:   [M] token IDs
            parents_flat: [M] parent indices in the flat list (-1 = root)
            budget:       target number of non-root nodes
            sample_token: root token

        Returns:
            (selected_tokens, selected_parents_in_selected_set)
            selected_tokens: [budget+1] including root at position 0
            selected_parents_in_selected_set: [budget] parent indices within selected set
        """
        M = scores_flat.shape[0]
        device = scores_flat.device

        # Score contribution = exp(log_prob)
        contributions = torch.exp(scores_flat)

        # Sort by contribution descending
        sorted_idx = torch.argsort(contributions, descending=True)

        # Track: old_index -> new_index in selected set (-1 = not selected)
        old_to_new = [-1] * M
        selected_old_indices = []  # in order of addition

        for idx in sorted_idx:
            idx_i = int(idx.item())
            if old_to_new[idx_i] != -1:
                continue  # already selected as ancestor

            # Trace ancestors: find path to root
            ancestors = []
            curr = idx_i
            while curr >= 0:
                if old_to_new[curr] != -1:
                    break  # ancestor already selected
                ancestors.append(curr)
                curr = int(parents_flat[curr].item())

            # Check budget
            if len(selected_old_indices) + len(ancestors) <= budget:
                # Select ancestors bottom-up (root first, then down)
                for anc in reversed(ancestors):
                    old_to_new[anc] = len(selected_old_indices)
                    selected_old_indices.append(anc)

        # Build selected token list (root + selected)
        selected_tokens = [sample_token]
        selected_parents_new = []
        for old_idx in selected_old_indices:
            selected_tokens.append(token_flat[old_idx])
            # Parent in old indexing
            old_parent = int(parents_flat[old_idx].item())
            if old_parent < 0:
                selected_parents_new.append(0)  # parent is root
            else:
                selected_parents_new.append(old_to_new[old_parent] + 1)  # +1 for root offset

        selected_tokens = torch.stack(selected_tokens)
        selected_parents_new = torch.tensor(selected_parents_new, dtype=torch.long, device=device)

        return selected_tokens, selected_parents_new

    @torch.no_grad()
    def topK_genrate(self, hidden_states, input_ids, head, logits_processor):

        input_ids = input_ids.to(hidden_states.device)
        total_tokens = self.total_tokens
        # DDD: use ddd_max_depth if enabled, otherwise use fixed depth
        depth = self.ddd_max_depth if self.use_ddd else self.depth
        top_k = self.top_k

        sample_token = input_ids[:, -1]

        scores_list = []
        parents_list = []
        ss_token = []
        ddd_call_checks = []
        ddd_early_stopped = False
        ddd_early_stop_step = None

        input_ids = input_ids[:, 1:]
        input_ids = input_ids.to(hidden_states.device)

        len_posi = input_ids.shape[1]
        self.reset()

        # with Timer("draft many"):
        if hasattr(self, "stable_kv") and self.stable_kv is not None:
            kv_len = self.stable_kv[0][0].shape[2]
            out_hidden, past_key_values = self(hidden_states, input_ids=input_ids[:, kv_len:],
                                               past_key_values=self.stable_kv, use_cache=True)
        else:
            out_hidden, past_key_values = self(hidden_states, input_ids=input_ids, use_cache=True)
        self.stable_kv = past_key_values
        last_hidden = out_hidden[:, -1]

        # last_headout = head(last_hidden)
        last_headout = self.lm_head(self.norm(last_hidden))

        last_p = self.logsoftmax(last_headout)
        top = torch.topk(last_p, top_k, dim=-1)
        topk_index, topk_p = top.indices, top.values
        scores = topk_p[0]
        scores_list.append(scores[None])
        parents_list.append(torch.zeros(1, dtype=torch.long, device=scores.device))
        if self.config.vocab_size==self.config.draft_vocab_size:
            ss_token.append(topk_index)
            input_ids = topk_index
        else:
            ss_token.append(topk_index+self.d2t[topk_index])
            input_ids = topk_index+self.d2t[topk_index]
        input_hidden = last_hidden[None].repeat(1, top_k, 1)
        tree_mask = self.tree_mask_init
        topk_cs_index = torch.arange(top_k, device=self.embed_tokens.weight.device)

        # 4
        for i in range(depth):
            self.tree_mask = tree_mask
            position_ids = len_posi + self.position_ids
            # with Timer("draft one"):
            out_hidden, past_key_values = self(input_hidden, input_ids=input_ids, past_key_values=past_key_values,
                                               position_ids=position_ids, use_cache=True)
            len_posi += 1

            # with Timer("sort1"):
            bias1 = top_k if i > 0 else 0
            bias2 = max(0, i - 1)
            bias = 1 + top_k ** 2 * bias2 + bias1
            parents = (topk_cs_index + bias)
            parents_list.append(parents)

            last_headout = self.lm_head(self.norm(out_hidden[0]))
            last_p = self.logsoftmax(last_headout)

            top = torch.topk(last_p, top_k, dim=-1)
            topk_index, topk_p = top.indices, top.values

            cu_scores = topk_p + scores[:, None]

            topk_cs = torch.topk(cu_scores.view(-1), top_k, dim=-1)
            topk_cs_index, topk_cs_p = topk_cs.indices, topk_cs.values
            scores = topk_cs_p

            out_ids = topk_cs_index // top_k
            input_hidden = out_hidden[:, out_ids]

            input_ids = topk_index.view(-1)[topk_cs_index][None]

            if self.config.vocab_size == self.config.draft_vocab_size:
                ss_token.append(topk_index)
            else:
                input_ids = input_ids + self.d2t[input_ids]
                ss_token.append(topk_index+self.d2t[topk_index])
            scores_list.append(cu_scores)
            tree_mask = torch.cat((tree_mask[:, :, out_ids], self.tree_mask_init), dim=3)

            # ---- DDD check: evaluate beam confidence at specified steps ----
            if self.use_ddd and (i + 1) in self.ddd_check_steps:
                self._ddd_stats["total_checks"] += 1
                # H = logsumexp(scores): log of total probability mass of top-k beams
                H = torch.logsumexp(scores, dim=0).item()
                check_record = {"step": i + 1, "H": H}
                ddd_call_checks.append(check_record)
                self._ddd_stats["checked_H"].append(check_record)
                if H < self.ddd_threshold:
                    self._ddd_stats["early_stops"] += 1
                    ddd_early_stopped = True
                    ddd_early_stop_step = i + 1
                    self._ddd_stats["early_stop_steps"].append(i + 1)
                    break
            # ---- end DDD check ----


        # ---- DDD: record actual depth ----
        actual_depth = len(ss_token)
        budget_tokens = total_tokens
        if self.use_ddd and self.use_ddd_dynamic_budget:
            max_actual_depth = max(1, self.ddd_max_depth + 1)
            scaled_budget = math.ceil(total_tokens * actual_depth / max_actual_depth)
            budget_tokens = max(self.ddd_min_budget, scaled_budget)
            budget_tokens = min(total_tokens, budget_tokens)

        if self.use_ddd:
            self._ddd_stats["calls"] += 1
            self._ddd_stats["depths"].append(actual_depth)
            self._ddd_stats["actual_depths"].append(actual_depth)
            self._ddd_stats["budget_tokens"].append(budget_tokens)
            self._ddd_stats["call_records"].append({
                "actual_depth": actual_depth,
                "budget_tokens": budget_tokens,
                "early_stopped": ddd_early_stopped,
                "early_stop_step": ddd_early_stop_step,
                "checks": ddd_call_checks,
            })
        # ---- end DDD ----

        scores_list = torch.cat(scores_list, dim=0).view(-1)
        ss_token_list = torch.cat(ss_token, dim=0).view(-1)

        tree_diff_record = None
        if hasattr(self, "_tree_diff_records"):
            parents_flat_for_diff = torch.cat(parents_list, dim=0).long()
            baseline_indices_for_diff = torch.sort(
                torch.topk(scores_list, budget_tokens, dim=-1).indices
            ).values

            def _candidate_depth_for_diff(candidate_index):
                depth_for_diff = 1
                parent_ref = int(parents_flat_for_diff[candidate_index // top_k].item())
                while parent_ref > 0:
                    candidate_index = parent_ref - 1
                    depth_for_diff += 1
                    parent_ref = int(parents_flat_for_diff[candidate_index // top_k].item())
                return depth_for_diff

            def _depth_hist_for_diff(indices):
                hist = {}
                for idx in indices.detach().cpu().tolist():
                    depth_key = str(_candidate_depth_for_diff(int(idx)))
                    hist[depth_key] = hist.get(depth_key, 0) + 1
                return dict(sorted(hist.items(), key=lambda item: int(item[0])))

            tree_diff_record = {
                "candidate_count": int(scores_list.shape[0]),
                "budget": int(budget_tokens),
                "top_k": int(top_k),
                "depth": int(depth),
                "baseline_indices": [int(x) for x in baseline_indices_for_diff.detach().cpu().tolist()],
                "baseline_depth_hist": _depth_hist_for_diff(baseline_indices_for_diff),
            }

        # ---- OPT-Tree: over-expand + greedy selection ----
        if self.use_opt_tree:
            over_N = max(budget_tokens + 1, int(budget_tokens * self.opt_expand_factor))
            over_N = min(over_N, scores_list.shape[0] - 1)  # cap at available candidates

            # 1) Select over_N nodes by score → intermediate tree
            top_over = torch.topk(scores_list, over_N, dim=-1)
            over_indices = top_over.indices
            over_indices = torch.sort(over_indices).values
            over_tokens = ss_token_list[over_indices]
            over_tokens = torch.cat((sample_token, over_tokens), dim=0)
            over_parents = torch.cat(parents_list, dim=0)[over_indices // top_k].long()

            # Build intermediate tree mask
            mask_inter = torch.searchsorted(over_indices, over_parents - 1, right=False)
            mask_inter[over_parents == 0] = -1
            mask_inter = mask_inter + 1
            mask_inter_list = mask_inter.tolist()
            tree_inter = torch.eye(over_N + 1).bool()
            tree_inter[:, 0] = True
            for i in range(over_N):
                tree_inter[i + 1].add_(tree_inter[mask_inter_list[i]])

            # 2) Precompute direct parent pointers for intermediate tree [O(N)]
            #    Use position_ids: depth[i] - depth[parent] = 1
            inter_depth = torch.sum(tree_inter, dim=1) - 1  # [over_N+1]
            inter_parents = [0] * (over_N + 1)
            for i in range(1, over_N + 1):
                d_i = int(inter_depth[i].item())
                # Direct parent = first ancestor with depth = d_i - 1, scanning upward
                for p in range(i - 1, -1, -1):
                    if tree_inter[i, p] and int(inter_depth[p].item()) == d_i - 1:
                        inter_parents[i] = p
                        break

            # 3) Compute contributions and apply greedy selection
            orig_scores_for_over = scores_list[over_indices]
            contributions = torch.exp(orig_scores_for_over)

            sorted_idx = torch.argsort(contributions, descending=True)
            selected_over_positions = set()
            budget = budget_tokens

            for idx in sorted_idx:
                idx_i = int(idx.item())
                inter_i = idx_i + 1
                if inter_i in selected_over_positions:
                    continue

                # Trace ancestors upward using precomputed parent pointers
                ancestors = []
                curr = inter_i
                while curr > 0:
                    if curr in selected_over_positions:
                        break
                    ancestors.append(curr)
                    curr = inter_parents[curr]

                if len(selected_over_positions) + len(ancestors) <= budget:
                    for anc in ancestors:
                        selected_over_positions.add(anc)

            # 4) Build final tree
            selected_sorted = sorted(selected_over_positions)
            inter_to_final = {0: 0}
            for fi, inter_pos in enumerate(selected_sorted):
                inter_to_final[inter_pos] = fi + 1

            final_N = len(selected_sorted)
            draft_tokens_list = [sample_token]
            draft_parents_final = []
            opt_selected_indices = [int(over_indices[inter_pos - 1].item()) for inter_pos in selected_sorted]

            for inter_pos in selected_sorted:
                orig_idx = int(over_indices[inter_pos - 1].item())
                tok = ss_token_list[orig_idx]
                draft_tokens_list.append(tok)
                parent_final = inter_to_final.get(inter_parents[inter_pos], 0)
                draft_parents_final.append(parent_final)

            draft_tokens = torch.cat([t.reshape(1) for t in draft_tokens_list], dim=0)

            # Build final tree mask
            tree_mask = torch.eye(final_N + 1).bool()
            tree_mask[:, 0] = True
            for i in range(final_N):
                tree_mask[i + 1].add_(tree_mask[draft_parents_final[i]])

            tree_position_ids = torch.sum(tree_mask, dim=1) - 1
            tree_mask = tree_mask.float()[None, None]
            draft_tokens = draft_tokens[None]

            total_tokens_actual = final_N
            mask_index_actual = torch.tensor(draft_parents_final, dtype=torch.long,
                                             device=scores_list.device)

            if tree_diff_record is not None:
                baseline_set = set(tree_diff_record["baseline_indices"])
                opt_set = set(opt_selected_indices)
                union_count = len(baseline_set | opt_set)
                overlap_count = len(baseline_set & opt_set)
                opt_indices_tensor = torch.tensor(opt_selected_indices, device=scores_list.device)
                tree_diff_record.update({
                    "opt_expand_factor": float(self.opt_expand_factor),
                    "over_budget": int(over_N),
                    "opt_final_count": int(final_N),
                    "opt_over_indices": [int(x) for x in over_indices.detach().cpu().tolist()],
                    "opt_selected_indices": opt_selected_indices,
                    "opt_depth_hist": _depth_hist_for_diff(opt_indices_tensor),
                    "overlap_count": int(overlap_count),
                    "union_count": int(union_count),
                    "jaccard": float(overlap_count / union_count) if union_count else 1.0,
                    "baseline_only_count": int(len(baseline_set - opt_set)),
                    "opt_only_count": int(len(opt_set - baseline_set)),
                })
                self._tree_diff_records.append(tree_diff_record)

        else:
            # ---- Original EAGLE selection (unchanged) ----
            top_scores = torch.topk(scores_list, budget_tokens, dim=-1)
            top_scores_index = top_scores.indices
            top_scores_index = torch.sort(top_scores_index).values

            draft_tokens = ss_token_list[top_scores_index]
            draft_tokens = torch.cat((sample_token, draft_tokens), dim=0)

            draft_parents = torch.cat(parents_list, dim=0)[top_scores_index // top_k].long()
            mask_index = torch.searchsorted(top_scores_index, draft_parents - 1, right=False)
            mask_index[draft_parents == 0] = -1
            mask_index = mask_index + 1
            mask_index_list = mask_index.tolist()
            tree_mask = torch.eye(budget_tokens + 1).bool()
            tree_mask[:, 0] = True
            for i in range(budget_tokens):
                tree_mask[i + 1].add_(tree_mask[mask_index_list[i]])

            tree_position_ids = torch.sum(tree_mask, dim=1) - 1
            tree_mask = tree_mask.float()[None, None]
            draft_tokens = draft_tokens[None]

            total_tokens_actual = budget_tokens
            mask_index_actual = mask_index

        del parents_list, scores_list, ss_token, ss_token_list

        # with Timer("retrieve"):
        # For OPT-Tree, use actual token count and mask
        _N = total_tokens_actual
        _mask = mask_index_actual
        _mask_list = _mask.tolist()

        max_depth = torch.max(tree_position_ids) + 1
        noleaf_index = torch.unique(_mask).tolist()
        noleaf_num = len(noleaf_index) - 1
        leaf_num = _N - noleaf_num

        retrieve_indices = torch.zeros(leaf_num, max_depth.item(), dtype=torch.long) - 1
        retrieve_indices = retrieve_indices.tolist()

        rid = 0
        position_ids_list = tree_position_ids.tolist()

        for i in range(_N + 1):
            if i not in noleaf_index:
                cid = i
                depth = position_ids_list[i]
                for j in reversed(range(depth + 1)):
                    retrieve_indices[rid][j] = cid
                    cid = _mask_list[cid - 1]
                rid += 1

        if logits_processor is not None:
            maxitem = _N + 5

            def custom_sort(lst):
                # sort_keys=[len(list)]
                sort_keys = []
                for i in range(len(lst)):
                    sort_keys.append(lst[i] if lst[i] >= 0 else maxitem)
                return sort_keys

            retrieve_indices = sorted(retrieve_indices, key=custom_sort)

        retrieve_indices = torch.tensor(retrieve_indices, dtype=torch.long)
        del _mask, _mask_list, noleaf_index, noleaf_num, leaf_num, max_depth, rid
        tree_position_ids = tree_position_ids.to(hidden_states.device)

        return draft_tokens, retrieve_indices, tree_mask, tree_position_ids




import torch


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    config = EConfig.from_pretrained('config.json')
    model = Model(config, load_emb=False)
    print(model)
