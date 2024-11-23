import torch
import torch.nn as nn

from typing import Tuple, Optional


class SigLIPVisionConfig:

    def __init__(
        self,
        hidden_size: int = 768,
        intermediate_size: int = 3072,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 12,
        num_channels: int = 3,
        image_size: int = 224,
        patch_size: int = 16,
        layer_norm_eps: float = 1e-12,
        attention_dropout: float = 0.0,
        num_image_tokens: int = None,
        **kwargs,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.image_size = image_size
        self.patch_size = patch_size
        self.layer_norm_eps = layer_norm_eps
        self.attention_dropout = attention_dropout
        self.num_image_tokens = num_image_tokens


class SigLIPVisionEmbeddings(nn.Module):

    def __init__(self, config: SigLIPVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            padding="valid",  # "valid" means no padding.
        )

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)

        self.register_buffer(
            "position_ids",
            torch.arange(self.num_positions).expand((1, -1)),
            persistent=False,
        )

    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        _, _, height, width = (
            pixel_values.shape
        )  # (batch_size, channels, height, width)
        # convolve the image with the patch_size kernel with no overlap patches since stride=patch_size.
        # output of the convolution is (batch_size, hidden_size, num_patches_h, num_patches_w)
        # where num_patches_h = height // patch_size and num_patches_w = width // patch_size
        patch_embeddings = self.patch_embedding(pixel_values)
        # (batch_size, hidden_size, num_patches_h, num_patches_w) -> (batch_size, hidden_size, num_patches)
        # where num_patches = num_patches_h * num_patches_w
        embeddings = patch_embeddings.flatten(2)
        embeddings = embeddings.transpose(
            1, 2
        )  # (batch_size, num_patches, hidden_size)

        embeddings = embeddings + self.position_embeddings(self.position_ids)

        return embeddings  # (batch_size, num_patches, hidden_size)


class SigLIPMLP(nn.Module):

    def __init__(self, config: SigLIPVisionConfig):
        super().__init__()
        self.config = config
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # (batch_size, num_patches, hidden_size) -> (batch_size, num_patches, intermediate_size)
        hidden_states = self.fc1(hidden_states)
        # (batch_size, num_patches, intermediate_size)
        hidden_states = nn.functional.gelu(hidden_states, approximate="tanh")
        # (batch_size, num_patches, intermediate_size) -> (batch_size, num_patches, hidden_size)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class SigLIPAttention(nn.Module):

    def __init__(self, config: SigLIPVisionConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = (
            self.head_dim**-0.5
        )  # mentioned in the original Transformer paper, 1/sqrt(head_dim)
        self.dropout = config.attention_dropout

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(
        self, hidden_states: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # hidden_states: (batch_size, num_patches, hidden_size)
        batch_size, seq_len, _ = hidden_states.size()
        # (batch_size, num_patches, hidden_size) -> (batch_size, num_patches, hidden_size)
        query_states = self.q_proj(hidden_states)
        # (batch_size, num_patches, hidden_size) -> (batch_size, num_patches, hidden_size)
        key_states = self.k_proj(hidden_states)
        # (batch_size, num_patches, hidden_size) -> (batch_size, num_patches, hidden_size)
        value_states = self.v_proj(hidden_states)
        # (batch_size, num_patches, hidden_size) -> (batch_size, num_heads, num_patches, head_dim)
        query_states = query_states.view(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        # (batch_size, num_patches, hidden_size) -> (batch_size, num_heads, num_patches, head_dim)
        key_states = key_states.view(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        # (batch_size, num_patches, hidden_size) -> (batch_size, num_heads, num_patches, head_dim)
        value_states = value_states.view(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        # (batch_size, num_heads, num_patches, head_dim) -> (batch_size, num_heads, num_patches, num_patches)
        attn_weights = (
            torch.matmul(query_states, key_states.transpose(2, 3)) * self.scale
        )
        # assert attn_weights.size() == (batch_size, self.num_heads, seq_len, seq_len)
        if attn_weights.size() != (batch_size, self.num_heads, seq_len, seq_len):
            raise ValueError(
                f"Attention weights should have the shape "
                f"(batch_size, num_heads, num_patches, num_patches), but got "
                f"{attn_weights.size()}"
            )
        # apply softmax to the attention weights so that the values sum up to 1.
        # (batch_size, num_heads, num_patches, num_patches)
        attn_weights = nn.functional.softmax(
            attn_weights, dim=-1, dtype=torch.float32
        ).to(query_states.dtype)
        # apply dropout to the attention weights when training.
        attn_weights = nn.functional.dropout(
            attn_weights, p=self.dropout, training=self.training
        )
        # (batch_size, num_heads, num_patches, num_patches) -> (batch_size, num_heads, num_patches, head_dim)
        attn_output = torch.matmul(attn_weights, value_states)
        # assert attn_output.size() == (batch_size, self.num_heads, seq_len, self.head_dim)
        if attn_output.size() != (batch_size, self.num_heads, seq_len, self.head_dim):
            raise ValueError(
                f"Attention output should have the shape "
                f"(batch_size, num_heads, num_patches, head_dim), but got "
                f"{attn_output.size()}"
            )
        # (batch_size, num_heads, num_patches, head_dim) -> (batch_size, num_patches, num_heads, head_dim)
        attn_output = attn_output.transpose(1, 2).contiguous()
        # (batch_size, num_patches, num_heads, head_dim) -> (batch_size, num_patches, hidden_size)
        attn_output = attn_output.reshape(batch_size, seq_len, self.embed_dim)
        # (batch_size, num_patches, hidden_size) -> (batch_size, num_patches, hidden_size)
        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights


class SigLIPEncoderLayer(nn.Module):

    def __init__(self, config: SigLIPVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.self_attn = SigLIPAttention(config)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = SigLIPMLP(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)

    def forward(self, inputs_embeds: torch.Tensor) -> torch.Tensor:
        residual = inputs_embeds  # (batch_size, num_patches, hidden_size)
        # (batch_size, num_patches, hidden_size)
        hidden_states = self.layer_norm1(inputs_embeds)
        # (batch_size, num_patches, hidden_size)
        hidden_states, _ = self.self_attn(hidden_states=hidden_states)
        # (batch_size, num_patches, hidden_size)
        hidden_states = hidden_states + residual
        # (batch_size, num_patches, hidden_size)
        residual = hidden_states
        # (batch_size, num_patches, hidden_size)
        hidden_states = self.layer_norm2(hidden_states)
        # (batch_size, num_patches, hidden_size)
        hidden_states = self.mlp(hidden_states=hidden_states)
        # (batch_size, num_patches, hidden_size)
        hidden_states = hidden_states + residual
        # (batch_size, num_patches, hidden_size)
        return hidden_states


class SigLIPVisionEncoder(nn.Module):

    def __init__(self, config: SigLIPVisionConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList(
            [SigLIPEncoderLayer(config) for _ in range(config.num_hidden_layers)]
        )

    def forward(self, inputs_embeds: torch.Tensor) -> torch.Tensor:
        # (batch_size, num_patches, hidden_size)
        hidden_states = inputs_embeds

        for layer in self.layers:
            # (batch_size, num_patches, hidden_size) -> (batch_size, num_patches, hidden_size)
            hidden_states = layer(hidden_states=hidden_states)

        return hidden_states


class SigLIPVisionTransformer(nn.Module):

    def __init__(self, config: SigLIPVisionConfig):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size

        self.embeddings = SigLIPVisionEmbeddings(config)
        self.encoder = SigLIPVisionEncoder(config)
        self.post_layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)

    def forward(self, pixel_values) -> torch.Tensor:
        # (batch_size, channels, height, width) -> (batch_size, num_image_tokens, hidden_size)
        hidden_states = self.embeddings(pixel_values)
        last_hidden_state = self.encoder(inputs_embeds=hidden_states)
        last_hidden_state = self.post_layernorm(last_hidden_state)

        return last_hidden_state


class SigLIPVisionModel(nn.Module):

    def __init__(self, config: SigLIPVisionConfig):
        super().__init__()
        self.config = config
        self.vision_model = SigLIPVisionTransformer(config)

    def forward(self, pixel_values) -> Tuple:
        # (batch_size, channels, height, width) -> (batch_size, num_image_tokens, hidden_size)
        return self.vision_model(pixel_values=pixel_values)
