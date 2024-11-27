import os
import math
from typing import List, Tuple, Optional

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from siglip.model import SigLIPVisionConfig, SigLIPVisionModel


class KVCache:

    def __init__(self):
        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []

    def num_items(self) -> int:
        if len(self.key_cache) == 0:
            return 0

        else:
            # key_states: (batch_size, num_kv_heads, seq_len, head_dim)
            return self.key_cache[0].shape[-2]

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if len(self.key_cache) <= layer_idx:
            # create the KVCache for that layer.
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)

        else:
            # (batch_size, num_kv_heads, seq_len, head_dim)
            self.key_cache[layer_idx] = torch.cat(
                [self.key_cache[layer_idx], key_states], dim=-2
            )
            self.value_cache[layer_idx] = torch.cat(
                [self.value_cache[layer_idx], value_states], dim=-2
            )

        return self.key_cache[layer_idx], self.value_cache[layer_idx]


class GemmaConfig:

    def __init__(
        self,
        vocab_size: None,
        hidden_size: None,
        intermediate_size: None,
        num_hidden_layers: None,
        num_attention_heads: None,
        num_key_value_heads: None,
        head_dim: int = 256,
        max_position_embeddings: int = 8192,
        rms_norm_eps: float = 1e-6,
        rope_theta: float = 10000.0,
        attention_bias: bool = False,
        attention_dropout: float = 0.0,
        pad_token_id: int = None,
        **kwargs,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.pad_token_id = pad_token_id


class PaliGemmaConfig:

    def __init__(
        self,
        vision_config: None,
        text_config: None,
        ignore_index: int = -100,
        image_token_index: int = 256000,
        vocab_size: int = 257152,
        projection_dim: int = 2048,
        hidden_size: int = 2048,
        pad_token_id: int = None,
        **kwargs,
    ):
        super().__init__()
        self.ignore_index = ignore_index
        self.image_token_index = image_token_index
        self.projection_dim = projection_dim
        self.hidden_size = hidden_size
        self.pad_token_id = pad_token_id
        self.vocab_size = vocab_size
        self.is_encoder_decoder = False

        self.vision_config = SigLIPVisionConfig(**vision_config)

        self.text_config = text_config
        self.text_config = GemmaConfig(**text_config, pad_token_id=pad_token_id)
        self.vocab_size = self.text_config.vocab_size

        self.text_config.num_image_tokens = (
            self.vision_config.image_size // self.vision_config.patch_size
        ) ** 2
        self.vision_config.projection_dim = projection_dim


class GemmaRMSNorm(nn.Module):

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float())
        output = output * (1.0 + self.weight.float())

        return output.type_as(x)


class GemmaMLP(nn.Module):

    def __init__(self, config: GemmaConfig):
        super().__init__()
        self.config = config
        self.hidden_size = self.config.hidden_size
        self.intermediate_size = self.config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

    def forward(self, x):
        # (batch_size, seq_len, hidden_size) -> (batch_size, seq_len, intermediate_size)
        hidden_states = self.gate_proj(x)
        # (batch_size, seq_len, intermediate_size)
        hidden_states = nn.functional.gelu(hidden_states, approximate="tanh")
        # (batch_size, seq_len, hidden_size) -> (batch_size, seq_len, intermediate_size)
        hidden_states_up = self.up_proj(x)
        # (batch_size, seq_len, intermediate_size)
        hidden_states = hidden_states * hidden_states_up
        # (batch_size, seq_len, intermediate_size) -> (batch_size, seq_len, hidden_size)
        hidden_states = self.down_proj(hidden_states)

        return hidden_states


class GemmaRotaryEmbedding(nn.Module):

    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()
        # RoPE is applied to each head independently so it's head dim.
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        inv_freq = 1.0 / (
            self.base
            ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float() / self.dim)
        )
        self.register_buffer("inv_freq", tensor=inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x, position_ids, seq_len=None):
        # x: (batch_size, num_heads, seq_len, head_dim)
        self.inv_freq.to(x.device)
        # inv_freq_expanded: (batch_size, head_dim // 2, 1)
        inv_freq_expanded = (
            self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        )
        # (batch_size, 1, seq_len)
        position_ids_expanded = position_ids[:, None, :].float()
        device_type = x.device.type
        device_type = (
            device_type
            if isinstance(device_type, str) and device_type != "mps"
            else "cpu"
        )

        with torch.autocast(device_type=device_type, enabled=False):
            # (batch_size, head_dim // 2, 1) @ (batch_size, 1, seq_len) -> (batch_size, seq_len, head_dim//2)
            freqs = (
                inv_freq_expanded.float() @ position_ids_expanded.float()
            ).transpose(1, 2)
            # (batch_size, seq_len, head_dim)
            emb = torch.cat((freqs, freqs), dim=-1)
            # (batch_size, seq_len, head_dim)
            cos = emb.cos()
            sin = emb.sin()

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class GemmaAttention(nn.ModuleDict):

    def __init__(self, config: GemmaConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.attention_dropout = self.config.attention_dropout
        self.hidden_size = self.config.hidden_size
        self.num_heads = self.config.num_attention_heads
        self.head_dim = self.config.head_dim
        self.num_kv_heads = self.config.num_key_value_heads
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.max_position_embeddings = self.config.max_position_embeddings
        self.rope_theta = self.config.rope_theta
        self.is_causal = True

        assert self.hidden_size % self.num_heads == 0

        # hidden_size = 1024, n_heads = 8, head_dim = 1024/8 = 128, kv_heads = 1
        # Wq = [1024, 8 * 128]
        # Wk = [1024, 1 * 128]
        # Wv = [1024, 1 * 128]
        self.q_proj = nn.Linear(
            self.hidden_size,
            self.num_heads * self.head_dim,
            bias=self.config.attention_bias,
        )
        self.k_proj = nn.Linear(
            self.hidden_size,
            self.num_kv_heads * self.head_dim,
            bias=self.config.attention_bias,
        )
        self.v_proj = nn.Linear(
            self.hidden_size,
            self.num_kv_heads * self.head_dim,
            bias=self.config.attention_bias,
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim,
            self.hidden_size,
            bias=self.config.attention_bias,
        )

        self.rotary_emb = GemmaRotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )

    def _rotate_half(self, x):
        # [-x2, x1, -x4, x3, ...]
        # First half of the last dimension.
        x1 = x[..., : x.shape[-1] // 2]
        # Second half of the last dimension.
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def _apply_rotary_pos_emb(self, q, k, cos, sin, unsqueeze_dim=1):
        # add head dimension.
        cos = cos.unsqueeze(unsqueeze_dim)
        sin = sin.unsqueeze(unsqueeze_dim)

        # Formula mentioned in Eq: 34 of the RoFormer paper.
        q_embed = (q * cos) + (self._rotate_half(q) * sin)
        k_embed = (k * cos) + (self._rotate_half(k) * sin)

        # (batch_size, num_q_heads, seq_len, head_dim), (batch_size, num_kv_heads, seq_len, head_dim)
        return q_embed, k_embed

    def _repeat_kv(self, hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
        batch_size, n_kv_heads, seq_len, head_dim = hidden_states.shape

        if n_rep == 1:
            return hidden_states

        hidden_states = hidden_states[:, :, None, :, :].expand(
            batch_size, n_kv_heads, n_rep, seq_len, head_dim
        )

        return hidden_states.reshape(batch_size, n_kv_heads * n_rep, seq_len, head_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        kv_cache: Optional[KVCache] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        # (batch_size, seq_len, hidden_size)
        batch_size, s_len, _ = hidden_states.size()
        # (batch_size, seq_len, num_heads_q * head_dim)
        query_states = self.q_proj(hidden_states)
        # (batch_size, seq_len, num_kv_heads * head_dim)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # (batch_size, num_heads_q, seq_len, head_dim)
        query_states = query_states.view(
            batch_size, s_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        # (batch_size, num_kv_heads, seq_len, head_dim)
        key_states = key_states.view(
            batch_size, s_len, self.num_kv_heads, self.head_dim
        ).transpose(1, 2)
        value_states = value_states.view(
            batch_size, s_len, self.num_kv_heads, self.head_dim
        ).transpose(1, 2)

        # (batch_size, s_len, head_dim), (batch_size, s_len, head_dim)
        cos, sin = self.rotary_emb(
            value_states,
            position_ids,
            seq_len=None,
        )
        # (batch_size, num_heads_q, seq_len, head_dim), (batch_size, num_kv_heads, seq_len, head_dim)
        query_states, key_states = self._apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )

        if kv_cache is not None:
            key_states, value_states = kv_cache.update(
                key_states, value_states, self.layer_idx
            )

        key_states = self._repeat_kv(key_states, self.num_kv_groups)
        value_states = self._repeat_kv(value_states, self.num_kv_groups)

        # (batch_size, num_heads_q, seq_len_q, seq_len_kv)
        attn_weights = torch.matmul(
            query_states, key_states.transpose(2, 3)
        ) / math.sqrt(self.head_dim)

        assert attention_mask is not None
        # (batch_size, num_heads_q, seq_len_q, seq_len_kv)
        attn_weights = attn_weights + attention_mask

        # apply softmax to the attention weights so that the values sum up to 1.
        # (batch_size, num_heads_q, seq_len_q, seq_len_kv)
        attn_weights = nn.functional.softmax(
            attn_weights, dim=-1, dtype=torch.float32
        ).to(query_states.dtype)
        # apply dropout to the attention weights when training.
        attn_weights = nn.functional.dropout(
            attn_weights, p=self.attention_dropout, training=self.training
        )
        # (batch_size, num_heads_q, seq_len_q, seq_len_kv) -> (batch_size, num_heads_q, seq_len_q, head_dim)
        attn_output = torch.matmul(attn_weights, value_states)
        # assert attn_output.size() == (batch_size, self.num_heads, seq_len, self.head_dim)
        if attn_output.size() != (batch_size, self.num_heads, s_len, self.head_dim):
            raise ValueError(
                f"Attention output should have the shape "
                f"(batch_size, num_heads, seq_len, head_dim), but got "
                f"{attn_output.size()}"
            )

        # (batch_size, num_heads_q, seq_len_q, head_dim) -> (batch_size, seq_len_q, num_heads_q, head_dim)
        attn_output = attn_output.transpose(1, 2).contiguous()
        # (batch_size, seq_len_q, num_heads_q, head_dim) -> (batch_size, seq_len_q, num_heads_q * head_dim)
        attn_output = attn_output.view(batch_size, s_len, -1)
        # (batch_size, seq_len_q, hidden_size)
        attn_output = self.o_proj(attn_output)

        return attn_output, attn_weights


class GemmaDecoderLayer(nn.Module):

    def __init__(self, config: GemmaConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = GemmaAttention(config, layer_idx=layer_idx)
        self.mlp = GemmaMLP(config)

        self.input_layernorm = GemmaRMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = GemmaRMSNorm(
            self.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        kv_cache: Optional[KVCache] = None,
    ) -> Tuple[
        torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]
    ]:
        # (batch_size, seq_len, hidden_size)
        residual = hidden_states
        # (batch_size, seq_len, hidden_size)
        hidden_states = self.input_layernorm(hidden_states)
        # (batch_size, seq_len, hidden_size)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            kv_cache=kv_cache,
        )
        # (batch_size, seq_len, hidden_size)
        hidden_states = hidden_states + residual
        residual = hidden_states
        # (batch_size, seq_len, hidden_size)
        hidden_states = self.post_attention_layernorm(hidden_states)
        # (batch_size, seq_len, hidden_size)
        hidden_states = self.mlp(hidden_states)
        # (batch_size, seq_len, hidden_size)
        hidden_states = hidden_states + residual

        return hidden_states


class GemmaModel(nn.Module):

    def __init__(self, config: GemmaConfig):
        super().__init__()
        self.config = config
        self.padding_idx = self.config.pad_token_id

        self.embed_tokens = nn.Embedding(
            self.config.vocab_size, self.config.hidden_size, self.padding_idx
        )
        self.layers = nn.ModuleList(
            [
                GemmaDecoderLayer(config, layer_idx)
                for layer_idx in range(self.config.num_hidden_layers)
            ]
        )
        self.norm = GemmaRMSNorm(self.config.hidden_size, self.config.rms_norm_eps)

    def get_input_embeddings(self):
        return self.embed_tokens

    def forward(
        self,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        input_embeds: Optional[torch.FloatTensor] = None,
        kv_cache: Optional[KVCache] = None,
    ) -> torch.FloatTensor:
        # (batch_size, seq_len, hidden_size)
        hidden_states = input_embeds
        norma = torch.tensor(self.config.hidden_size**0.5, dtype=hidden_states.dtype)
        # (batch_size, seq_len, hidden_size)
        hidden_states = hidden_states * norma

        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                kv_cache=kv_cache,
            )
        # (batch_size, seq_len, hidden_size)
        hidden_states = self.norm(hidden_states)

        return hidden_states


class GemmaForCausalLM(nn.Module):

    def __init__(self, config: GemmaConfig):
        super().__init__()
        self.config = config
        self.model = GemmaModel(config)
        self.vocab_size = self.config.vocab_size
        self.lm_head = nn.Linear(self.config.hidden_size, self.vocab_size, bias=False)

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def tie_weights(self):
        self.lm_head.weight = self.model.embed_tokens.weight

    def forward(
        self,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        input_embeds: Optional[torch.FloatTensor] = None,
        kv_cache: Optional[KVCache] = None,
    ) -> Tuple:
        # input_embeds: (batch_size, seq_length, hidden_size)
        # outputs: (batch_size, seq_length, hidden_size)
        outputs = self.model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            input_embeds=input_embeds,
            kv_cache=kv_cache,
        )

        hidden_states = outputs
        logits = self.lm_head(hidden_states)
        logits = logits.float()

        return_data = {
            "logits": logits,
        }

        if kv_cache is not None:
            return_data["kv_cache"] = kv_cache

        return return_data


class PaliGemmaMultiModalProjector(nn.Module):

    def __init__(self, config: PaliGemmaConfig):
        super().__init__()
        self.linear = nn.Linear(
            config.vision_config.hidden_size, config.vision_config.projection_dim
        )

    def forward(self, image_features: torch.Tensor) -> torch.Tensor:
        # (batch_size, num_patches, hidden_size) -> (batch_size, num_patches, projection_dim)
        hidden_states = self.linear(image_features)

        return hidden_states


class PaliGemmaForConditionalGeneration(nn.Module):

    def __init__(self, config: PaliGemmaConfig):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.vision_tower = SigLIPVisionModel(config=config.vision_config)
        self.multi_modal_projector = PaliGemmaMultiModalProjector(config)

        language_model = GemmaForCausalLM(config.text_config)
        self.language_model = language_model

        self.pad_token_id = (
            self.config.pad_token_id if self.config.pad_token_id is not None else -1
        )

    def tie_weights(self):
        """Tie the weights between the input embeddings and the output embeddings."""

        return self.language_model.tie_weights()

    def _merge_input_ids_and_pixel_features(
        self,
        image_features: torch.Tensor,
        input_embeds: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        kv_cache: Optional[KVCache] = None,
    ):
        _, _, embed_size = image_features.shape
        batch_size, seq_length = input_ids.shape
        dtype, device = input_embeds.dtype, input_embeds.device

        # 1. Scale the image features to the hidden size:
        # (batch_size, num_patches, hidden_size)
        scaled_image_features = image_features / (self.config.hidden_size**0.5)

        # 2. Create the final input embeddings and masks:
        # (batch_size, seq_length, hidden_size)
        final_embeds = torch.zeros(
            batch_size, seq_length, embed_size, dtype=dtype, device=device
        )
        # (batch_size, seq_length)
        text_mask = (input_ids != self.config.image_token_index) & (
            input_ids != self.pad_token_id
        )
        # (batch_size, seq_length)
        image_mask = input_ids == self.config.image_token_index
        # (batch_size, seq_length)
        pad_mask = input_ids == self.pad_token_id

        # Expand the masks to the embedding dimension:
        # (batch_size, seq_length, hidden_size)
        text_mask_expanded = text_mask.unsqueeze(-1).expand(-1, -1, embed_size)
        image_mask_expanded = image_mask.unsqueeze(-1).expand(-1, -1, embed_size)
        pad_mask_expanded = pad_mask.unsqueeze(-1).expand(-1, -1, embed_size)

        # 3. Add the embeddings to the final embeddings:
        final_embeds = torch.where(text_mask_expanded, input_embeds, final_embeds)
        # using the masked_scatter function because the scaled_image_features size is different to final_embeds.
        final_embeds = final_embeds.masked_scatter(
            image_mask_expanded, scaled_image_features
        )
        final_embeds = torch.where(
            pad_mask_expanded, torch.zeros_like(final_embeds), final_embeds
        )

        # 4. Create the attention mask:
        q_len = input_embeds.size(1)

        if kv_cache is None or kv_cache.num_items() == 0:
            # since the input is not padded, we don't need to mask any tokens.
            # in the Gemma paper, figure 2 shows that all the tokens are not masked during the prefilling.
            causal_mask = torch.full(
                (batch_size, q_len, q_len), fill_value=0, dtype=dtype, device=device
            )

        else:
            assert q_len == 1
            kv_len = kv_cache.num_items() + q_len
            # since the input is not padded, the query token can attend all the previous tokens.
            causal_mask = torch.full(
                (batch_size, q_len, kv_len), fill_value=0, dtype=dtype, device=device
            )

        # add the head dimension to the mask.
        # (batch_size, q_len, kv_len) -> (batch_size, 1, q_len, kv_len) where 1 is the head dimension.
        causal_mask = causal_mask.unsqueeze(1)

        # 5. Create the position ids:
        if kv_cache is not None or kv_cache.num_items() > 0:
            # the position of the query token is the last token in the sequence.
            # it extracts the last token in the sequence.
            position_ids = attention_mask.cumsum(dim=-1)[:, -1]
            if position_ids.dim() == 1:
                position_ids = position_ids.unsqueeze(0)

        else:
            # create the position ids based on the attention mask with valid tokens.
            # for the masked tokens (mask = 0), the position ids are set to 1.
            position_ids = (
                (attention_mask.cumsum(dim=-1))
                .masked_fill((attention_mask == 0), 1)
                .to(device)
            )

        return final_embeds, causal_mask, position_ids

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        pixel_values: torch.FloatTensor = None,
        attention_mask: Optional[torch.LongTensor] = None,
        kv_cache: Optional[KVCache] = None,
    ) -> Tuple:
        assert torch.all(attention_mask == 1), "The input cannot be padded"

        # 1. Extract the input embeddings:
        # (batch_size, seq_length, hidden_size)
        input_embeds = self.language_model.get_input_embeddings()(input_ids)

        # 2. Merge the input embeddings with the vision features:
        # (batch_size, channels, height, width) -> (batch_size, num_patches, embed_size)
        selected_img_features = self.vision_tower(pixel_values.to(input_embeds.dtype))
        # (batch_size, num_patches, embed_size) -> (batch_size, num_patches, hidden_size)
        image_features = self.multi_modal_projector(selected_img_features)

        # 3. Merge the input embeddings with the vision features:
        # (batch_size, seq_length, hidden_size)
        input_embeds, attention_mask, position_ids = (
            self._merge_input_ids_and_pixel_features(
                image_features, input_embeds, input_ids, attention_mask, kv_cache
            )
        )

        # 4. Generate the output:
        outputs = self.language_model(
            input_embeds=input_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            kv_cache=kv_cache,
        )

        return outputs
