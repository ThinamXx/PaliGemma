import math
from typing import List, Tuple, Optional

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from siglip.model import SigLIPVisionConfig, SigLIPVisionModel


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
        super().__init__(**kwargs)
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
        super().__init__(**kwargs)
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


class PaliGemmaForConditionalGeneration(nn.Module):

    def __init__(self, config: PaliGemmaConfig):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.vision_model = SigLIPVisionModel(config=SigLIPVisionConfig)
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
        selected_img_features = self.vision_model(pixel_values.to(input_embeds.dtype))
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
