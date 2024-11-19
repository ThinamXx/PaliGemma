import torch
from torch import nn

import os
import json
import glob
import numpy as np
from safetensors import safe_open
from PIL import Image
from typing import List, Dict, Optional, Union, Tuple, Iterable

from transformers import AutoTokenizer
from model import PaliGemmaForConditionalGeneration, PaliGemmaConfig


IMAGENET_STANDARD_MEAN = [0.5, 0.5, 0.5]
IMAGENET_STANDARD_STD = [0.5, 0.5, 0.5]


def add_image_tokens_to_prompt(
    prefix_prompt: str,
    bos_token: str,
    image_seq_length: int,
    image_token: str,
):
    """Function to add image tokens to the input text.

    Quoting from the blog (https://huggingface.co/blog/paligemma#detailed-inference-process):
    "image_tokens" -> "bos_token" -> "prefix_prompt" -> "\n"

    From the paper it looks like the `\n` should be tokenized separately, but in the HF implementation this is not done.
    ref to HF implementation: https://github.com/huggingface/transformers/blob/7f79a97399bb52aad8460e1da2f36577d5dccfed/src/transformers/models/paligemma/processing_paligemma.py#L55-L73
    """

    input_text = f"{image_token * image_seq_length}{bos_token}{prefix_prompt}\n"

    return input_text


def resize(
    image: Image.Image,
    size: Tuple[int, int],
    resample: Image.Resampling = None,
    reducing_gap: Optional[int] = None,
) -> np.ndarray:
    """Function to resize the input image."""

    height, width = size
    resized_image = image.resize(
        (width, height), resample=resample, reducing_gap=reducing_gap
    )

    return resized_image


def rescale(
    image: np.ndarray,
    scale: float,
    dtype: np.dtype = np.float32,
) -> np.ndarray:
    """Function to rescale the pixel values of the image to [0, 1]."""

    rescaled_image = image * scale  # scale is 1.0 / 255.0
    rescaled_image = rescaled_image.astype(dtype)

    return rescaled_image


def normalize(
    image: np.ndarray,
    mean: Union[float, Iterable[float]],
    std: Union[float, Iterable[float]],
) -> np.ndarray:
    """Function to normalize the pixel values of the image."""

    mean = np.array(mean, dtype=image.dtype)
    std = np.array(std, dtype=image.dtype)
    image = (image - mean) / std

    return image


def process_images(
    images: List[Image.Image],
    size: Dict[str, int] = None,
    resample: Image.Resampling = None,
    rescale_factor: float = None,
    image_mean: Optional[Union[float, List[float]]] = None,
    image_std: Optional[Union[float, List[float]]] = None,
) -> List[np.ndarray]:
    """Process the input images.

    Args:
        images (List[Image.Image]): List of PIL images.
        size (Dict[str, int], optional): Desired size of the images. Defaults to None.
        resample (Image.Resampling, optional): Resampling filter. Defaults to None.
        rescale_factor (float, optional): Factor to rescale the pixel values. Defaults to None.
        image_mean (Optional[Union[float, List[float]]], optional): Normalization mean. Defaults to None.
        image_std (Optional[Union[float, List[float]]], optional): Normalization standard deviation. Defaults to None.

    Returns:
        List[np.ndarray]: List of processed images.
    """

    height, width = size[0], size[1]
    # Resize the image.
    images = [
        resize(image=image, size=(height, width), resample=resample) for image in images
    ]
    # Convert the image to numpy array.
    images = [np.array(image) for image in images]
    # Rescale the pixel values to [0, 1].
    images = [rescale(image, scale=rescale_factor) for image in images]
    # Normalize the pixel values to have zero mean and unit variance.
    images = [normalize(image, mean=image_mean, std=image_std) for image in images]
    # (height, width, channel) to (channel, height, width)
    images = [image.transpose(2, 0, 1) for image in images]

    return images


class PaliGemmaProcessor:
    """Class to process the input text and images for the PaliGemma model.

    Returns:
        Dict: A dictionary containing the pixel values of the images and the tokenized input text.
    """

    # Placeholder for image tokens that will be added to the input text tokens.
    IMAGE_TOKEN = "<image>"

    def __init__(self, tokenizer, num_image_tokens: int, image_size: int):
        super().__init__()
        self.image_seq_length = num_image_tokens
        self.image_size = image_size

        # Tokenizer: https://github.com/google-research/big_vision/blob/main/big_vision/configs/proj/paligemma/README.md#tokenizer
        tokens_to_add = {"additional_special_tokens": [self.IMAGE_TOKEN]}
        tokenizer.add_special_tokens(tokens_to_add)
        # Tokens for object detection.
        EXTRA_TOKENS = [f"<loc{i:04d}>" for i in range(1024)]
        # Tokens for image segmentation.
        EXTRA_TOKENS += [f"<seg{i:03d}>" for i in range(128)]
        tokenizer.add_special_tokens(EXTRA_TOKENS)
        self.image_token_id = tokenizer.convert_tokens_to_ids(self.IMAGE_TOKEN)
        tokenizer.add_bos_token = False
        tokenizer.add_eos_token = False

        self.tokenizer = tokenizer

    def __call__(
        self,
        text: List[str],
        images: List[Image.Image],
        padding: str = "longest",
        truncation: bool = True,
    ) -> dict:
        assert (
            len(images) == 1 and len(text) == 1
        ), f"Received {len(images)} images for {len(text)} prompt."

        pixel_values = process_images(
            images,
            size=(self.image_size, self.image_size),
            resample=Image.Resampling.BICUBIC,
            rescale_factor=1.0 / 255.0,
            image_mean=IMAGENET_STANDARD_MEAN,
            image_std=IMAGENET_STANDARD_STD,
        )

        pixel_values = np.stack(
            pixel_values, axis=0
        )  # (batch_size, channel, image_size, image_size)
        pixel_values = torch.tensor(pixel_values)  # convert to single array tensor.

        # Prepend self.image_seq_length, number of image tokens to the input text.
        input_strings = [
            add_image_tokens_to_prompt(
                prefix_prompt=prompt,
                bos_token=self.tokenizer.bos_token,
                image_seq_length=self.image_seq_length,
                image_token=self.IMAGE_TOKEN,
            )
            for prompt in text
        ]

        # Tokenize the input strings.
        inputs = self.tokenizer(
            input_strings,
            padding=padding,
            truncation=truncation,
            return_tensors="pt",
        )

        return_data = {"pixel_values": pixel_values, **inputs}

        return return_data


def load_hf_model(
    model_path: str,
    device: str,
) -> Tuple[PaliGemmaForConditionalGeneration, AutoTokenizer]:
    # 1. Load the Tokenizer.
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="right")
    assert tokenizer.padding_side == "right"

    # 2. Load the safetensors.
    safetensors_files = glob.glob(os.path.join(model_path, "*.safetensors"))
    tensors = {}
    for safetensor_file in safetensors_files:
        with safe_open(safetensor_file, framework="pt", device="cpu") as f:
            for key in f.keys():
                tensors[key] = f.get_tensor(key)

    # 3. Load model's config.
    with open(os.path.json(model_path, "config.json"), "r") as f:
        model_config_file = json.load(f)
        config = PaliGemmaConfig(**model_config_file)

    # 4. Create the model.
    model = PaliGemmaForConditionalGeneration(config=config).to(device)
    model = model.load_state_dict(tensors, strict=False)

    model.tie_weights()

    return (model, tokenizer)
