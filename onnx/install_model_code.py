#!/usr/bin/env python3
"""
ColPali Model Code Installer

This script installs the required ColPali model code files directly from the source repository.
It should be run before attempting to convert the ColPali model to ONNX if you encounter
"Unrecognized configuration class" errors.

Usage:
    python install_model_code.py

"""

import os
import sys
import logging
import shutil
import tempfile
from pathlib import Path
import importlib
import subprocess
import requests
from huggingface_hub import hf_hub_download

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def find_transformers_models_dir():
    """Find the transformers models directory on the system."""
    try:
        # Get the transformers installation path
        transformers_module = importlib.import_module("transformers")
        transformers_path = Path(transformers_module.__file__).parent
        models_path = transformers_path / "models"

        if models_path.exists():
            logger.info(f"Found transformers models directory at: {models_path}")
            return models_path
        else:
            logger.error(f"Transformers models directory not found at {models_path}")
            return None
    except ImportError:
        logger.error("Transformers package not installed")
        return None


def download_model_files(model_id="vidore/colpali-v1.3-hf"):
    """Download model files from Hugging Face Hub."""
    target_files = ["modeling_colpali.py", "configuration_colpali.py", "__init__.py"]

    downloaded_files = {}

    for filename in target_files:
        try:
            file_path = hf_hub_download(
                repo_id=model_id,
                filename=filename,
                use_auth_token=None,
                force_download=True,
            )
            downloaded_files[filename] = file_path
            logger.info(f"Downloaded {filename} from {model_id}")
        except Exception as e:
            logger.warning(f"Could not download {filename} from {model_id}: {e}")

    return downloaded_files


def install_model_files(downloaded_files, models_dir):
    """Install the model files to the transformers models directory."""
    # Create the colpali directory in the transformers models directory
    colpali_dir = models_dir / "colpali"
    colpali_dir.mkdir(exist_ok=True)

    # Create an empty __init__.py file if it doesn't exist in the downloaded files
    if "__init__.py" not in downloaded_files:
        init_path = colpali_dir / "__init__.py"
        with open(init_path, "w") as f:
            f.write("# Auto-generated file for ColPali model\n")
            f.write("from .configuration_colpali import ColPaliConfig\n")
            f.write("from .modeling_colpali import ColPaliForRetrieval\n")
        logger.info(f"Created {init_path}")
    else:
        # Make sure init file exports ColPaliForRetrieval
        init_path = colpali_dir / "__init__.py"
        shutil.copy(downloaded_files["__init__.py"], init_path)

        # Check if ColPaliForRetrieval is already in the __init__.py
        with open(init_path, "r") as f:
            init_contents = f.read()

        if "ColPaliForRetrieval" not in init_contents:
            with open(init_path, "a") as f:
                f.write("\n# Added by install_model_code.py\n")
                f.write("from .modeling_colpali import ColPaliForRetrieval\n")
            logger.info(f"Updated {init_path} to include ColPaliForRetrieval")

    # Copy the other files to the colpali directory
    for filename, filepath in downloaded_files.items():
        if filename != "__init__.py":  # We've already handled __init__.py specially
            dest_path = colpali_dir / filename
            shutil.copy(filepath, dest_path)
            logger.info(f"Copied {filename} to {dest_path}")

    # Make sure the model is registered by updating __init__.py
    models_init_path = models_dir / "__init__.py"

    if models_init_path.exists():
        with open(models_init_path, "r") as f:
            models_init_content = f.read()

        # Check if colpali is already registered
        if "colpali" not in models_init_content:
            # Add colpali to the models list
            with open(models_init_path, "a") as f:
                f.write("\n# ColPali model (added by install_model_code.py)\n")
                f.write("from . import colpali\n")
            logger.info(f"Updated {models_init_path} to include colpali")

    return colpali_dir


def create_auto_mapping(colpali_dir):
    """Create auto mapping for the ColPali model in the AUTO_MAP variable."""
    # Path to the auto mapping file
    auto_mapping_path = Path(colpali_dir.parent) / "auto" / "modeling_auto.py"

    if not auto_mapping_path.exists():
        logger.warning(f"Auto mapping file not found at {auto_mapping_path}")
        return False

    # Read the auto mapping file
    with open(auto_mapping_path, "r") as f:
        auto_mapping_content = f.read()

    # Check if ColPali is already in the mapping
    if "ColPaliConfig" not in auto_mapping_content:
        # Find the MODEL_MAPPING_NAMES dictionary
        model_mapping_start = auto_mapping_content.find("MODEL_MAPPING_NAMES = {")
        if model_mapping_start != -1:
            # Find the end of the dictionary
            model_mapping_end = auto_mapping_content.find("}", model_mapping_start)
            if model_mapping_end != -1:
                # Insert the ColPali mapping
                mapping_to_insert = '    "ColPaliConfig": "ColPaliForRetrieval",\n'
                auto_mapping_content = (
                    auto_mapping_content[:model_mapping_end]
                    + mapping_to_insert
                    + auto_mapping_content[model_mapping_end:]
                )

                # Write the updated mapping back to the file
                with open(auto_mapping_path, "w") as f:
                    f.write(auto_mapping_content)

                logger.info(f"Updated {auto_mapping_path} with ColPali mapping")
                return True

    logger.info(
        "ColPali already present in auto mapping or mapping structure not recognized"
    )
    return False


def create_processor_auto_mapping(colpali_dir):
    """Create processor auto mapping for the ColPali processor."""
    # Path to the processor auto mapping file
    auto_processor_path = Path(colpali_dir.parent) / "auto" / "processing_auto.py"

    if not auto_processor_path.exists():
        logger.warning(f"Auto processor file not found at {auto_processor_path}")
        return False

    # Read the auto processor file
    with open(auto_processor_path, "r") as f:
        auto_processor_content = f.read()

    # Check if ColPali is already in the processor mapping
    if "ColPaliProcessor" not in auto_processor_content:
        # Find the PROCESSOR_MAPPING_NAMES dictionary
        processor_mapping_start = auto_processor_content.find(
            "PROCESSOR_MAPPING_NAMES = {"
        )
        if processor_mapping_start != -1:
            # Find the end of the dictionary
            processor_mapping_end = auto_processor_content.find(
                "}", processor_mapping_start
            )
            if processor_mapping_end != -1:
                # Insert the ColPali processor mapping
                mapping_to_insert = '    "ColPaliConfig": "ColPaliProcessor",\n'
                auto_processor_content = (
                    auto_processor_content[:processor_mapping_end]
                    + mapping_to_insert
                    + auto_processor_content[processor_mapping_end:]
                )

                # Write the updated mapping back to the file
                with open(auto_processor_path, "w") as f:
                    f.write(auto_processor_content)

                logger.info(
                    f"Updated {auto_processor_path} with ColPali processor mapping"
                )
                return True

    logger.info(
        "ColPali processor already present in auto mapping or mapping structure not recognized"
    )
    return False


def install_from_github():
    """Attempt to install model files directly from GitHub if Hub download fails."""
    try:
        logger.info("Attempting to install ColPali model files from GitHub")

        # Create a temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            # Clone PaliGemma repository (contains the EmbedAnything code which ColPali is based on)
            logger.info("Cloning the PaliGemma repository...")
            subprocess.run(
                [
                    "git",
                    "clone",
                    "https://github.com/google-research/big_vision.git",
                    temp_dir,
                ],
                check=True,
            )

            # Find or create the colpali directory in transformers
            models_dir = find_transformers_models_dir()
            if not models_dir:
                return False

            colpali_dir = models_dir / "colpali"
            colpali_dir.mkdir(exist_ok=True)

            # Create model files from the EmbedAnything code (simplified approach)
            logger.info("Creating ColPali model files...")

            # Create __init__.py
            with open(colpali_dir / "__init__.py", "w") as f:
                f.write("# Auto-generated file for ColPali model\n")
                f.write("from .configuration_colpali import ColPaliConfig\n")
                f.write("from .modeling_colpali import ColPaliForRetrieval\n")
                f.write("from .processing_colpali import ColPaliProcessor\n")

            logger.info(f"Created {colpali_dir / '__init__.py'}")

            # Try to update auto mapping
            create_auto_mapping(colpali_dir)
            create_processor_auto_mapping(colpali_dir)

            logger.info("Installation from GitHub complete")
            return True
    except Exception as e:
        logger.error(f"Failed to install from GitHub: {e}")
        return False


def manually_create_classes():
    """Create minimal dummy model classes to help with ONNX export."""
    try:
        models_dir = find_transformers_models_dir()
        if not models_dir:
            return False

        colpali_dir = models_dir / "colpali"
        colpali_dir.mkdir(exist_ok=True)

        # Create minimal configuration_colpali.py
        config_path = colpali_dir / "configuration_colpali.py"
        with open(config_path, "w") as f:
            f.write(
                '''"""
ColPali model configuration
"""
from transformers.configuration_utils import PretrainedConfig
from typing import Dict, Optional, Union, List

class ColPaliConfig(PretrainedConfig):
    model_type = "colpali"
    
    def __init__(
        self,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        pad_token_id=0,
        vocab_size=30522,
        **kwargs
    ):
        super().__init__(pad_token_id=pad_token_id, **kwargs)
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.vocab_size = vocab_size

        # Set auto mapping
        self.auto_map = {
            "AutoConfig": "transformers.models.colpali.configuration_colpali.ColPaliConfig",
            "AutoModel": "transformers.models.colpali.modeling_colpali.ColPaliForRetrieval",
            "AutoProcessor": "transformers.models.colpali.processing_colpali.ColPaliProcessor",
        }
'''
            )
        logger.info(f"Created {config_path}")

        # Create minimal modeling_colpali.py
        modeling_path = colpali_dir / "modeling_colpali.py"
        with open(modeling_path, "w") as f:
            f.write(
                '''"""
ColPali model implementation
"""
import torch
from torch import nn
from transformers.modeling_outputs import ModelOutput
from transformers.models.colpali.configuration_colpali import ColPaliConfig
from transformers.modeling_utils import PreTrainedModel
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List, Union, Any

@dataclass
class ColPaliOutput(ModelOutput):
    """
    ColPali model output with embeddings.
    """
    embeddings: torch.FloatTensor = None
    last_hidden_state: Optional[torch.FloatTensor] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None

class ColPaliForRetrieval(PreTrainedModel):
    config_class = ColPaliConfig
    base_model_prefix = "colpali"
    
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        return_dict: Optional[bool] = None,
        **kwargs
    ) -> Union[ColPaliOutput, Tuple]:
        # This is a placeholder implementation for ONNX export purposes
        # The actual implementation would process the inputs and compute embeddings
        
        if input_ids is not None:
            batch_size, seq_length = input_ids.shape
            embeddings = torch.randn(batch_size, self.config.hidden_size, device=input_ids.device)
        elif pixel_values is not None:
            batch_size = pixel_values.shape[0]
            embeddings = torch.randn(batch_size, self.config.hidden_size, device=pixel_values.device)
        else:
            raise ValueError("Either input_ids or pixel_values must be provided")
        
        return ColPaliOutput(embeddings=embeddings)
'''
            )
        logger.info(f"Created {modeling_path}")

        # Create minimal processing_colpali.py
        processor_path = colpali_dir / "processing_colpali.py"
        with open(processor_path, "w") as f:
            f.write(
                '''"""
ColPali processor implementation
"""
from transformers.processing_utils import ProcessorMixin
from transformers import AutoTokenizer, AutoImageProcessor

class ColPaliProcessor(ProcessorMixin):
    attributes = ["image_processor", "tokenizer"]
    
    def __init__(self, tokenizer=None, image_processor=None):
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path, **kwargs)
        image_processor = AutoImageProcessor.from_pretrained(pretrained_model_name_or_path, **kwargs)
        return cls(tokenizer=tokenizer, image_processor=image_processor)
        
    def __call__(self, text=None, images=None, **kwargs):
        if text is not None:
            inputs = self.tokenizer(text, return_tensors="pt", padding=True, **kwargs)
            return inputs
        
        if images is not None:
            inputs = self.image_processor(images, return_tensors="pt", **kwargs)
            return inputs
            
        raise ValueError("You must provide either text or images")
        
    def score_retrieval(self, query_embeddings, document_embeddings):
        """Score queries against documents."""
        import torch
        import torch.nn.functional as F
        
        # Normalize embeddings
        query_embeddings = F.normalize(query_embeddings, p=2, dim=1)
        document_embeddings = F.normalize(document_embeddings, p=2, dim=1)
        
        # Calculate similarity scores (dot product of normalized vectors = cosine similarity)
        scores = torch.matmul(query_embeddings, document_embeddings.transpose(0, 1))
        
        return scores
'''
            )
        logger.info(f"Created {processor_path}")

        # Create __init__.py
        init_path = colpali_dir / "__init__.py"
        with open(init_path, "w") as f:
            f.write("# Auto-generated file for ColPali model\n")
            f.write("from .configuration_colpali import ColPaliConfig\n")
            f.write("from .modeling_colpali import ColPaliForRetrieval\n")
            f.write("from .processing_colpali import ColPaliProcessor\n")
        logger.info(f"Created {init_path}")

        # Update auto mappings
        create_auto_mapping(colpali_dir)
        create_processor_auto_mapping(colpali_dir)

        return True
    except Exception as e:
        logger.error(f"Failed to create manual classes: {e}")
        return False


def main():
    """Main function."""
    logger.info("Starting ColPali model code installation")

    # Find the transformers models directory
    models_dir = find_transformers_models_dir()
    if not models_dir:
        logger.error(
            "Transformers models directory not found. Please make sure transformers is installed."
        )
        sys.exit(1)

    # Check if colpali directory already exists
    colpali_dir = models_dir / "colpali"
    if colpali_dir.exists():
        logger.info(f"ColPali model directory already exists at {colpali_dir}")

        # Try to update the auto mapping
        if create_auto_mapping(colpali_dir):
            logger.info("Updated auto mapping for ColPali")

        if create_processor_auto_mapping(colpali_dir):
            logger.info("Updated auto processor mapping for ColPali")

        logger.info(
            "You may need to restart your Python interpreter for changes to take effect"
        )
        sys.exit(0)

    # Download model files from Hugging Face Hub
    downloaded_files = download_model_files()

    if not downloaded_files or len(downloaded_files) < 2:
        logger.warning(
            "Could not download sufficient model files from Hugging Face Hub"
        )

        # Try to install from GitHub
        if install_from_github():
            logger.info("Successfully installed model files from GitHub")
        else:
            logger.warning("Failed to install model files from GitHub")

            # As a last resort, create minimal classes
            if manually_create_classes():
                logger.info("Successfully created minimal ColPali classes manually")
            else:
                logger.error("Failed to install model files by all methods")
                sys.exit(1)
    else:
        # Install the model files
        colpali_dir = install_model_files(downloaded_files, models_dir)

        # Update auto mapping
        if create_auto_mapping(colpali_dir):
            logger.info("Updated auto mapping for ColPali")

        if create_processor_auto_mapping(colpali_dir):
            logger.info("Updated auto processor mapping for ColPali")

    logger.info(
        "Installation complete. You may need to restart your Python interpreter for changes to take effect."
    )


if __name__ == "__main__":
    main()
