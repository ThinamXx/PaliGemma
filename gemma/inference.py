import torch
import fire
from PIL import Image

from utils import PaliGemmaProcessor, load_hf_model
from model import PaliGemmaForConditionalGeneration, KVCache


def move_inputs_to_device(model_inputs: dict, device: str):
    model_inputs = {k: v.to(device) for k, v in model_inputs.items()}

    return model_inputs


def get_model_input(
    processor: PaliGemmaProcessor,
    prompt: str,
    image_file_path: str,
    device: str,
):
    image = Image.open(image_file_path)
    images = [image]
    prompts = [prompt]
    model_inputs = processor(text=prompts, images=images)
    model_inputs = move_inputs_to_device(model_inputs=model_inputs, device=device)

    return model_inputs


def test_inference(
    model: PaliGemmaForConditionalGeneration,
    procesor: PaliGemmaProcessor,
    device: str,
    prompt: str,
    image_file_path: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    do_sample: bool,
):
    # TODO:
    pass


def main(
    model_path: str = None,
    prompt: str = None,
    image_file_path: str = None,
    max_tokens: int = 100,
    temperature: float = 0.8,
    top_p: float = 0.9,
    do_sample: bool = False,
    only_cpu: bool = False,
):
    device = "cpu"
    if not only_cpu:
        if torch.cuda.is_available():
            device = "cuda"

        elif torch.backends.mps.is_available():
            device = "mps"

    print(f"Device in use {device}")
    print(f"Loading the model.")
    model, tokenizer = load_hf_model(model_path, device)
    model = model.to(device).eval()

    num_image_tokens = model.config.vision_config.num_image_tokens
    image_size = model.config.vision_config.image_size
    procesor = PaliGemmaProcessor(
        tokenizer=tokenizer, num_image_tokens=num_image_tokens, image_size=image_size
    )

    print(f"Running the inference.")
    with torch.no_grad():
        test_inference(
            model,
            procesor,
            device,
            prompt,
            image_file_path,
            max_tokens,
            temperature,
            top_p,
            do_sample,
        )


if __name__ == "__main__":
    fire.Fire(main)
