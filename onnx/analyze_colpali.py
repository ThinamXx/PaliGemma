"""
ColPali Model Architecture Analyzer

This script analyzes the architecture of the vidore/colpali-v1.3-hf model
to understand its structure for ONNX conversion.

Usage:
    python analyze_colpali.py --model_id vidore/colpali-v1.3-hf

"""

import argparse
import os
import json
import torch
from transformers import AutoConfig, AutoModel, AutoProcessor, AutoTokenizer
from typing import Dict, List, Tuple, Any, Optional
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze ColPali model architecture")
    parser.add_argument(
        "--model_id",
        type=str,
        default="vidore/colpali-v1.3-hf",
        help="Model ID on HuggingFace Hub or local path",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="colpali_analysis.json",
        help="File to save analysis results",
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        default=True,
        help="Trust remote code when loading the model",
    )
    return parser.parse_args()


def analyze_inputs_outputs(model, processor):
    """Analyze model inputs and outputs using sample data."""
    logger.info("Analyzing model inputs and outputs...")

    # Generate sample data
    sample_image = torch.rand(1, 3, 384, 384)
    sample_text = "This is a sample document text"

    # Process inputs with processor
    try:
        processor_inputs = None
        if processor is not None:
            logger.info("Processing inputs with processor...")
            try:
                processor_inputs = processor(
                    images=sample_image, text=sample_text, return_tensors="pt"
                )
                logger.info(f"Processor input keys: {list(processor_inputs.keys())}")
                for k, v in processor_inputs.items():
                    if hasattr(v, "shape"):
                        logger.info(f"  {k}: shape={v.shape}, dtype={v.dtype}")
            except Exception as e:
                logger.warning(
                    f"Error using processor with both image and text: {str(e)}"
                )
                try:
                    # Try image only
                    processor_inputs = processor(
                        images=sample_image, return_tensors="pt"
                    )
                    logger.info(
                        f"Processor (image-only) input keys: {list(processor_inputs.keys())}"
                    )
                except Exception as e:
                    logger.warning(f"Error using processor with image only: {str(e)}")
                    try:
                        # Try text only
                        processor_inputs = processor(
                            text=sample_text, return_tensors="pt"
                        )
                        logger.info(
                            f"Processor (text-only) input keys: {list(processor_inputs.keys())}"
                        )
                    except Exception as e:
                        logger.warning(
                            f"Error using processor with text only: {str(e)}"
                        )
    except Exception as e:
        logger.error(f"Error processing sample data: {str(e)}")
        processor_inputs = None

    # Try to run model with processed inputs
    try:
        logger.info("Running model forward pass...")
        model.eval()
        with torch.no_grad():
            if processor_inputs is not None:
                outputs = model(**processor_inputs)
                logger.info("Model successfully ran with processor inputs")
            else:
                # Try with basic inputs
                outputs = model(sample_image)
                logger.info("Model successfully ran with basic image input")
    except Exception as e:
        logger.error(f"Error running model forward pass: {str(e)}")
        outputs = None

    # Analyze outputs
    if outputs is not None:
        if hasattr(outputs, "keys"):
            logger.info(f"Output keys: {list(outputs.keys())}")
            for k, v in outputs.items():
                if hasattr(v, "shape"):
                    logger.info(f"  {k}: shape={v.shape}, dtype={v.dtype}")
        else:
            logger.info(f"Output: {type(outputs)}")
            if hasattr(outputs, "shape"):
                logger.info(f"  shape={outputs.shape}, dtype={outputs.dtype}")


def analyze_model_architecture(model):
    """Analyze model architecture."""
    logger.info("Analyzing model architecture...")

    # Get model parameters count
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")

    # Analyze model structure
    model_structure = {}
    for name, module in model.named_children():
        module_info = {
            "type": str(type(module).__name__),
            "parameters": sum(p.numel() for p in module.parameters()),
        }

        # Get submodules for complex modules
        if hasattr(module, "named_children"):
            submodules = {}
            for subname, submodule in module.named_children():
                submodules[subname] = {
                    "type": str(type(submodule).__name__),
                    "parameters": sum(p.numel() for p in submodule.parameters()),
                }
            if submodules:
                module_info["submodules"] = submodules

        model_structure[name] = module_info

    return {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "structure": model_structure,
    }


def analyze_config(config):
    """Analyze model configuration."""
    logger.info("Analyzing model configuration...")

    config_dict = config.to_dict()

    # Extract key configuration details
    key_config = {
        "model_type": config_dict.get("model_type", "Unknown"),
        "architectures": config_dict.get("architectures", []),
        "hidden_size": config_dict.get("hidden_size", None),
        "num_hidden_layers": config_dict.get("num_hidden_layers", None),
        "num_attention_heads": config_dict.get("num_attention_heads", None),
        "vocab_size": config_dict.get("vocab_size", None),
    }

    return {"key_config": key_config, "full_config": config_dict}


def main():
    args = parse_args()

    # Load config, model, and processor
    logger.info(f"Loading model configuration from {args.model_id}...")
    try:
        config = AutoConfig.from_pretrained(
            args.model_id, trust_remote_code=args.trust_remote_code
        )
    except Exception as e:
        logger.error(f"Error loading configuration: {str(e)}")
        config = None

    logger.info(f"Loading model from {args.model_id}...")
    try:
        model = AutoModel.from_pretrained(
            args.model_id, trust_remote_code=args.trust_remote_code
        )
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        model = None

    logger.info(f"Loading processor from {args.model_id}...")
    try:
        processor = AutoProcessor.from_pretrained(
            args.model_id, trust_remote_code=args.trust_remote_code
        )
    except Exception as e:
        logger.warning(f"Error loading processor: {str(e)}")
        logger.info("Trying to load tokenizer instead...")
        try:
            processor = AutoTokenizer.from_pretrained(
                args.model_id, trust_remote_code=args.trust_remote_code
            )
        except Exception as e:
            logger.warning(f"Error loading tokenizer: {str(e)}")
            processor = None

    # Collect analysis data
    analysis_data = {
        "model_id": args.model_id,
        "config": analyze_config(config) if config else None,
        "architecture": analyze_model_architecture(model) if model else None,
    }

    # Analyze inputs and outputs
    if model and processor:
        analyze_inputs_outputs(model, processor)

    # Save analysis data
    try:
        with open(args.output_file, "w") as f:
            json.dump(analysis_data, f, indent=2)
        logger.info(f"Analysis data saved to {args.output_file}")
    except Exception as e:
        logger.error(f"Error saving analysis data: {str(e)}")
        logger.info("Analysis data (partial):")
        logger.info(json.dumps(analysis_data, indent=2))


if __name__ == "__main__":
    main()
