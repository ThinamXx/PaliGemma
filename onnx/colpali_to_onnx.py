"""
ColPali to ONNX Converter

This script converts the vidore/colpali-v1.3-hf model (or alternatively, vidore/colqwen2-v1.0-merged)
to ONNX format using PyTorch's built-in ONNX export functionality with additional optimizations.

Usage:
    python colpali_to_onnx.py --model_id vidore/colpali-v1.3-hf --output_dir colpali_onnx

"""

import argparse
import os
import json
import logging
import warnings
import importlib
from pathlib import Path
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer, AutoProcessor
from transformers.utils import HUGGINGFACE_CO_RESOLVE_ENDPOINT
from huggingface_hub import hf_hub_download
from optimum.exporters.onnx import main_export, OnnxConfig
from optimum.exporters.onnx.base import ConfigBehavior
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Convert ColPali model to ONNX format")
    parser.add_argument(
        "--model_id",
        type=str,
        default="vidore/colpali-v1.3-hf",
        help="Model ID on HuggingFace Hub or local path",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="colpali_onnx",
        help="Directory to save ONNX model",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=15,
        help="ONNX opset version",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use for conversion (default: cuda if available, else cpu)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for conversion",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["float32", "float16"],
        default="float32",
        help="Data type for conversion",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-5,
        help="Absolute tolerance for validating ONNX model",
    )
    parser.add_argument(
        "--optimize",
        action="store_true",
        default=True,
        help="Optimize ONNX model after conversion",
    )
    parser.add_argument(
        "--quantize",
        action="store_true",
        help="Quantize ONNX model to INT8",
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        default=True,
        help="Trust remote code when loading the model",
    )
    parser.add_argument(
        "--input_shapes",
        type=str,
        default=None,
        help="JSON string with input shapes dictionary",
    )
    return parser.parse_args()


def get_device(device_arg):
    """Determine device based on argument and availability."""
    if device_arg is not None:
        return device_arg

    if torch.cuda.is_available():
        return "cuda:0"
    else:
        return "cpu"


def load_specific_model_class(model_id, config, trust_remote_code=True):
    """
    Dynamically load the appropriate model class based on the configuration.
    This handles models like ColPali that aren't in the standard AutoModel mapping.
    """
    logger.info(f"Loading specific model class for {config.__class__.__name__}")

    try:
        # Try to directly import the ColPaliForRetrieval class based on the HF documentation
        logger.info("Trying to import ColPaliForRetrieval directly")
        from transformers import ColPaliForRetrieval

        logger.info("Successfully imported ColPaliForRetrieval")
        model = ColPaliForRetrieval.from_pretrained(
            model_id, config=config, trust_remote_code=trust_remote_code
        )
        logger.info("Successfully loaded ColPaliForRetrieval model")
        return model
    except (ImportError, AttributeError) as e:
        logger.warning(f"Failed to import ColPaliForRetrieval: {str(e)}")

    try:
        # First, try to inspect the config to get the correct model class
        if hasattr(config, "auto_map") and config.auto_map:
            logger.info(f"Found auto_map in config: {config.auto_map}")
            if "AutoModel" in config.auto_map:
                model_class_path = config.auto_map["AutoModel"]
                logger.info(f"Found model class path: {model_class_path}")

                # Parse the module and class name
                module_name, class_name = model_class_path.rsplit(".", 1)
                logger.info(f"Importing {class_name} from {module_name}")

                # Import the module and get the class
                module = importlib.import_module(module_name)
                ModelClass = getattr(module, class_name)
                logger.info(f"Successfully imported {ModelClass}")

                # Create the model instance
                return ModelClass.from_pretrained(
                    model_id, config=config, trust_remote_code=trust_remote_code
                )
    except Exception as e:
        logger.warning(f"Failed to load model using auto_map: {str(e)}")

    # If auto_map approach fails, try common model naming patterns
    try:
        # Try to infer model class from config class name
        config_class_name = config.__class__.__name__  # e.g., "ColPaliConfig"
        if config_class_name.endswith("Config"):
            model_class_name = (
                config_class_name[:-6] + "ForRetrieval"
            )  # Remove "Config" and add "ForRetrieval"
            logger.info(f"Trying to load model class: {model_class_name}")

            # Try to find the module path
            config_module = (
                config.__class__.__module__
            )  # e.g., "transformers.models.colpali.configuration_colpali"
            model_module = config_module.replace("configuration_", "modeling_")
            logger.info(f"Looking for model in module: {model_module}")

            try:
                module = importlib.import_module(model_module)
                ModelClass = getattr(module, model_class_name)
                logger.info(f"Successfully imported {ModelClass}")

                return ModelClass.from_pretrained(
                    model_id, config=config, trust_remote_code=trust_remote_code
                )
            except (ImportError, AttributeError) as e:
                logger.warning(f"Failed to import from {model_module}: {str(e)}")
    except Exception as e:
        logger.warning(f"Failed to infer model class from config: {str(e)}")

    # As a last resort, try to directly access Hugging Face model code
    try:
        logger.info("Trying to access model code directly from Hugging Face Hub")

        # Find main model file
        repo_id = model_id
        if repo_id.startswith("/"):
            repo_id = model_id.replace("/", "", 1)

        # Try to find relevant model files in the repo
        model_files = ["modeling_colpali.py", "modeling.py", "model.py"]

        model_file = None
        for filename in model_files:
            try:
                model_file = hf_hub_download(
                    repo_id=repo_id,
                    filename=filename,
                    use_auth_token=None,
                    force_download=True,
                )
                logger.info(f"Found model file: {model_file}")
                break
            except Exception as e:
                logger.warning(f"File {filename} not found: {str(e)}")

        if model_file:
            logger.info(f"Loading model code from {model_file}")
            # We'll use from_pretrained from the model file
            # Import the module
            import sys
            import importlib.util

            spec = importlib.util.spec_from_file_location("model_module", model_file)
            model_module = importlib.util.module_from_spec(spec)
            sys.modules["model_module"] = model_module
            spec.loader.exec_module(model_module)

            # Look for model classes in the module
            model_class_names = ["ColPaliForRetrieval", "ColPaliModel", "Model"]
            for class_name in model_class_names:
                if hasattr(model_module, class_name):
                    ModelClass = getattr(model_module, class_name)
                    logger.info(f"Found model class {class_name} in {model_file}")
                    return ModelClass.from_pretrained(
                        model_id, config=config, trust_remote_code=trust_remote_code
                    )
    except Exception as e:
        logger.warning(f"Failed to load model from Hub: {str(e)}")

    # If all else fails
    raise ValueError(
        f"Could not load model class for {config.__class__.__name__}. "
        "Try installing the specific module for this model."
    )


class CustomOnnxConfig(OnnxConfig):
    """Custom ONNX configuration for ColPali model."""

    @property
    def inputs(self):
        """Define model inputs for ONNX conversion."""
        return {
            "pixel_values": {
                0: "batch_size",
                1: "num_channels",
                2: "height",
                3: "width",
            },
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
        }

    @property
    def outputs(self):
        """Define model outputs for ONNX conversion."""
        return {
            "last_hidden_state": {
                0: "batch_size",
                1: "sequence_length",
                2: "hidden_size",
            }
        }

    def generate_dummy_inputs(
        self, batch_size=1, sequence_length=32, height=384, width=384, num_channels=3
    ):
        """Generate dummy inputs for the model."""
        return {
            "pixel_values": torch.rand(batch_size, num_channels, height, width),
            "input_ids": torch.randint(0, 100, (batch_size, sequence_length)),
            "attention_mask": torch.ones(batch_size, sequence_length, dtype=torch.long),
        }

    def patch_model_for_export(self, model, **kwargs):
        """Apply any necessary model patching before export."""
        # No patching required for now
        return model


def export_model_to_onnx(model, args, config=None, processor=None, tokenizer=None):
    """Export model to ONNX format using PyTorch's built-in export."""
    device = get_device(args.device)
    logger.info(f"Using device: {device}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Move model to specified device and set eval mode
    model = model.to(device)
    model.eval()

    # Convert to specified dtype
    if args.dtype == "float16":
        model = model.half()

    # Get input names and shapes
    input_shapes = {}
    if args.input_shapes:
        try:
            input_shapes = json.loads(args.input_shapes)
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse input_shapes: {args.input_shapes}")

    # Create dummy inputs
    logger.info("Creating dummy inputs...")
    batch_size = args.batch_size
    sequence_length = input_shapes.get("sequence_length", 64)
    height = input_shapes.get("height", 384)
    width = input_shapes.get("width", 384)

    # Try to determine processor requirements
    try:
        if processor is not None:
            logger.info("Using processor to create input tensors...")
            # Sample text for processing
            sample_texts = ["sample text"] * batch_size

            # Create a blank image for processing
            if hasattr(processor, "image_processor"):
                logger.info("Using image processor component...")
                # Try to get image processor requirements
                sample_image = torch.rand(3, height, width)
                processed = processor.image_processor(
                    images=[sample_image.numpy()] * batch_size, return_tensors="pt"
                )
                pixel_values = processed.pixel_values.to(device)
            else:
                # Fallback to standard image shape
                pixel_values = torch.rand(batch_size, 3, height, width).to(device)

            # Process text inputs if the model supports it
            if hasattr(processor, "tokenizer"):
                logger.info("Using tokenizer component...")
                text_inputs = processor.tokenizer(
                    sample_texts,
                    padding="max_length",
                    max_length=sequence_length,
                    truncation=True,
                    return_tensors="pt",
                )
                input_ids = text_inputs.input_ids.to(device)
                attention_mask = text_inputs.attention_mask.to(device)

                # Complete input dict
                input_dict = {
                    "pixel_values": pixel_values,
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                }
            else:
                # Image-only model
                input_dict = {"pixel_values": pixel_values}

            # Convert to tuple for onnx export
            dummy_inputs = tuple(input_dict.values())
            input_names = list(input_dict.keys())
        else:
            # Fallback to basic inputs
            logger.info("No processor available, using fallback dummy inputs...")
            pixel_values = torch.rand(batch_size, 3, height, width).to(device)
            input_ids = torch.randint(0, 100, (batch_size, sequence_length)).to(device)
            attention_mask = torch.ones(
                batch_size, sequence_length, dtype=torch.long
            ).to(device)

            if args.dtype == "float16":
                pixel_values = pixel_values.half()

            dummy_inputs = (pixel_values, input_ids, attention_mask)
            input_names = ["pixel_values", "input_ids", "attention_mask"]
    except Exception as e:
        logger.warning(f"Error creating processor inputs: {str(e)}")
        # Fallback to even more basic input
        logger.info("Using minimal input with only pixel_values...")
        pixel_values = torch.rand(batch_size, 3, height, width).to(device)
        if args.dtype == "float16":
            pixel_values = pixel_values.half()
        dummy_inputs = (pixel_values,)
        input_names = ["pixel_values"]

    # Define output names
    output_names = ["last_hidden_state"]

    # Set dynamic axes
    dynamic_axes = {}
    for input_name in input_names:
        dynamic_axes[input_name] = {0: "batch_size"}
        if "ids" in input_name or "mask" in input_name:
            dynamic_axes[input_name][1] = "sequence_length"
        elif "pixel" in input_name:
            dynamic_axes[input_name][2] = "height"
            dynamic_axes[input_name][3] = "width"

    for output_name in output_names:
        dynamic_axes[output_name] = {0: "batch_size", 1: "sequence_length"}

    # Export model to ONNX
    onnx_path = os.path.join(args.output_dir, "model.onnx")
    logger.info(f"Exporting model to ONNX: {onnx_path}")

    try:
        with torch.no_grad():
            torch.onnx.export(
                model,
                dummy_inputs,
                onnx_path,
                input_names=input_names,
                output_names=output_names,
                dynamic_axes=dynamic_axes,
                opset_version=args.opset,
                do_constant_folding=True,
                verbose=False,
            )
    except Exception as e:
        logger.error(f"ONNX export failed: {str(e)}")

        # Try with simpler inputs if the export fails
        logger.info("Trying ONNX export with simplified inputs...")
        try:
            pixel_values = torch.rand(batch_size, 3, height, width).to(device)
            if args.dtype == "float16":
                pixel_values = pixel_values.half()

            with torch.no_grad():
                torch.onnx.export(
                    model,
                    (pixel_values,),
                    onnx_path,
                    input_names=["pixel_values"],
                    output_names=["output"],
                    dynamic_axes={
                        "pixel_values": {0: "batch_size", 2: "height", 3: "width"},
                        "output": {0: "batch_size"},
                    },
                    opset_version=args.opset,
                    do_constant_folding=True,
                )
        except Exception as e:
            logger.error(f"Simplified ONNX export also failed: {str(e)}")
            raise RuntimeError("Failed to export model to ONNX format")

    # Verify the ONNX model
    logger.info("Verifying ONNX model...")
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)

    # Save model configuration
    if config is not None:
        config.save_pretrained(args.output_dir)

    # Save tokenizer
    if tokenizer is not None:
        tokenizer.save_pretrained(args.output_dir)

    # Save processor
    if processor is not None:
        processor.save_pretrained(args.output_dir)

    # Optimize ONNX model if requested
    if args.optimize:
        try:
            import onnxruntime as ort
            from onnxruntime.transformers import optimizer

            logger.info("Optimizing ONNX model...")
            optimized_model_path = os.path.join(args.output_dir, "model_optimized.onnx")

            # Try with basic optimization without specific parameters
            try:
                opt_options = optimizer.OptimizationOptions()
                opt_model = optimizer.optimize_model(
                    onnx_path,
                    "bert",  # Using BERT as a base optimization type
                    optimization_options=opt_options,
                )
                opt_model.save_model_to_file(optimized_model_path)
            except Exception as e:
                logger.warning(f"Advanced optimization failed: {str(e)}")
                logger.info("Trying basic optimization...")

                # Fallback to basic optimization with onnxruntime
                session_options = ort.SessionOptions()
                session_options.graph_optimization_level = (
                    ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                )
                session_options.optimized_model_filepath = optimized_model_path
                _ = ort.InferenceSession(onnx_path, session_options)

            # Replace original model with optimized version if file exists
            if os.path.exists(optimized_model_path):
                os.replace(optimized_model_path, onnx_path)
                logger.info(f"Optimized model saved to {onnx_path}")
        except Exception as e:
            logger.warning(f"Model optimization failed: {str(e)}")

    # Quantize model if requested
    if args.quantize:
        try:
            logger.info("Quantizing ONNX model to INT8...")
            quantized_model_path = os.path.join(args.output_dir, "model_quantized.onnx")

            quantize_dynamic(
                onnx_path, quantized_model_path, weight_type=QuantType.QInt8
            )

            # Replace original model with quantized version
            os.replace(quantized_model_path, onnx_path)
            logger.info(f"Quantized model saved to {onnx_path}")
        except Exception as e:
            logger.warning(f"Model quantization failed: {str(e)}")

    logger.info(f"ONNX export complete. Model saved to {onnx_path}")
    return onnx_path


def try_optimum_export(args):
    """Try to export using Optimum's built-in functionality."""
    try:
        logger.info("Trying export with Optimum's main_export function...")
        main_export(
            model_name_or_path=args.model_id,
            output=args.output_dir,
            task="feature-extraction",
            opset=args.opset,
            device=get_device(args.device),
            trust_remote_code=args.trust_remote_code,
        )
        return True
    except Exception as e:
        logger.warning(f"Optimum export failed: {str(e)}")
        return False


def main():
    args = parse_args()
    device = get_device(args.device)
    logger.info(f"Starting conversion process for {args.model_id}")
    logger.info(f"Using device: {device}")

    # Load model configuration
    logger.info(f"Loading model configuration for {args.model_id}...")
    config = AutoConfig.from_pretrained(
        args.model_id, trust_remote_code=args.trust_remote_code
    )

    # Try to load tokenizer and processor
    logger.info("Loading tokenizer and processor...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_id, trust_remote_code=args.trust_remote_code
        )
        logger.info("Tokenizer loaded successfully")
    except Exception as e:
        logger.warning(f"Failed to load tokenizer: {str(e)}")
        tokenizer = None

    try:
        processor = AutoProcessor.from_pretrained(
            args.model_id, trust_remote_code=args.trust_remote_code
        )
        logger.info("Processor loaded successfully")
    except Exception as e:
        logger.warning(f"Failed to load processor: {str(e)}")
        processor = None

    # Try the optimum export first
    logger.info("Attempting export with Optimum...")
    if try_optimum_export(args):
        logger.info("Successfully converted model using Optimum.")
        return

    # If optimum fails, fall back to manual export
    logger.info("Falling back to manual export...")

    # Load model - use the dynamic loader instead of AutoModel
    logger.info(f"Loading model from {args.model_id}...")
    try:
        model = load_specific_model_class(
            args.model_id, config, trust_remote_code=args.trust_remote_code
        )
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")

        # Last resort: try direct import for ColPali
        if "colpali" in args.model_id.lower():
            try:
                logger.info("Attempting direct import of ColPali model...")
                # Try to directly import the ColPali model class
                from transformers.models.colpali.modeling_colpali import ColPaliModel

                model = ColPaliModel.from_pretrained(
                    args.model_id,
                    config=config,
                    trust_remote_code=args.trust_remote_code,
                )
                logger.info("ColPali model loaded successfully through direct import")
            except Exception as e2:
                logger.error(f"Direct import failed: {str(e2)}")
                raise RuntimeError(
                    f"All model loading attempts failed: {str(e)}, {str(e2)}"
                )
        else:
            raise RuntimeError(f"Failed to load model: {str(e)}")

    # Export model to ONNX
    export_model_to_onnx(model, args, config, processor, tokenizer)

    logger.info(f"Model successfully converted and saved to {args.output_dir}")
    logger.info(f"ONNX model path: {os.path.join(args.output_dir, 'model.onnx')}")


if __name__ == "__main__":
    main()
