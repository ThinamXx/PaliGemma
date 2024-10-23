import math
from typing import List, Tuple, Optional

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from siglip.model import SigLIPVisionConfig, SigLIPVisionModel


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
            self.merge_input_ids_and_pixel_features(
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
