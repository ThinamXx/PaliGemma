import torch
import fire
from PIL import Image

from utils import PaliGemmaProcessor, load_hf_model
from model import PaliGemmaForConditionalGeneration, KVCache


def move_inputs_to_device(model_inputs: dict, device: str):
    model_inputs = {k: v.to(device) for k, v in model_inputs.items()}

    return model_inputs


def sample_top_p(probs: torch.Tensor, p: float):
    # (batch_size, vocab_size)
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    # (batch_size, vocab_size)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    # (batch_size, vocab_size)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    # redistribute the probs so that they sum to 1.
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    # sample a token.
    next_token = torch.multinomial(probs_sort, num_samples=1)
    # get the token position from the vocab.
    next_token = torch.gather(probs_idx, -1, next_token)

    return next_token


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
    processor: PaliGemmaProcessor,
    device: str,
    prompt: str,
    image_file_path: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    do_sample: bool,
):
    model_inputs = get_model_input(processor, prompt, image_file_path, device)
    input_ids = model_inputs["input_ids"]
    attention_mask = model_inputs["attention_mask"]
    pixel_values = model_inputs["pixel_values"]

    kv_cache = KVCache()

    # Generate the next tokens until the stop token.
    stop_token = processor.tokenizer.eos_token_id
    generated_tokens = []

    for _ in range(max_tokens):
        model_outputs = model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            kv_cache=kv_cache,
        )

        kv_cache = model_outputs["kv_cache"]
        next_token_logits = model_outputs["logits"][:, -1, :]

        if do_sample:
            next_token_logits = torch.softmax(next_token_logits / temperature, dim=-1)
            next_token = sample_top_p(next_token_logits, top_p)
        else:
            next_token = torch.argmax(next_token_logits, keepdim=True)

        assert next_token.size() == (1, 1)

        # remove batch dimension.
        next_token = next_token.squeeze(0)
        generated_tokens.append(next_token)

        if next_token.item() == stop_token:
            # break if stop token is reached.
            break

        # Append the next token for next iteration.
        input_ids = next_token.unsqueeze(-1)
        attention_mask = torch.cat(
            [attention_mask, torch.ones((1, 1), device=input_ids.device)], dim=-1
        )

    generated_tokens = torch.cat(generated_tokens, dim=-1)
    decoded_tokens = processor.tokenizer.decode(
        generated_tokens, skip_special_tokens=True
    )

    return prompt + decoded_tokens


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
