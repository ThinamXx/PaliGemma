import torch
from torch import nn

import numpy as np
from PIL import Image
from typing import List, Dict, Optional, Union, Tuple


IMAGENET_STANDARD_MEAN = [0.5, 0.5, 0.5]
IMAGENET_STANDARD_STD = [0.5, 0.5, 0.5]


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
