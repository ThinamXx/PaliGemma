from typing import Tuple, Optional

import torch
import torch.nn as nn


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
        self.position_embeddings = nn.Embedding(self.num_positions, self.embed_dim)

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
        last_hidden_state = self.encoder(input_embeds=hidden_states)
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
