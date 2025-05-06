"""
ColPali Document Retrieval using ONNX

This script demonstrates how to use the converted ColPali ONNX model for real-world
document retrieval tasks. It includes functionality to:
1. Index a folder of document images
2. Search the index with text queries
3. Retrieve and rank the most relevant documents

Usage:
    # Index documents
    python colpali_doc_retrieval.py --mode index --model_dir colpali_onnx --doc_dir ./documents --index_path doc_index.pkl

    # Search the index
    python colpali_doc_retrieval.py --mode search --model_dir colpali_onnx --index_path doc_index.pkl --query "invoice for shoes"

"""

import argparse
import os
import pickle
import json
import numpy as np
import torch
import onnxruntime
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from transformers import AutoProcessor, AutoTokenizer
import logging
from typing import Dict, List, Tuple, Any, Optional

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="ColPali Document Retrieval with ONNX")
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["index", "search"],
        help="Mode: 'index' to create document index, 'search' to query the index",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="colpali_onnx",
        help="Directory containing ONNX model and processor",
    )
    parser.add_argument(
        "--doc_dir",
        type=str,
        help="Directory containing document images to index",
    )
    parser.add_argument(
        "--index_path",
        type=str,
        default="doc_index.pkl",
        help="Path to save/load document index",
    )
    parser.add_argument(
        "--query",
        type=str,
        help="Text query for document search",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="Number of top results to return",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda"],
        default="cpu",
        help="Device to use for inference",
    )
    return parser.parse_args()


def load_onnx_model(model_path, device="cpu"):
    """Load ONNX model with ONNX Runtime."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"ONNX model not found at {model_path}")

    # Set up ONNX Runtime session
    sess_options = onnxruntime.SessionOptions()
    sess_options.graph_optimization_level = (
        onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    )

    # Configure provider based on device
    if (
        device == "cuda"
        and "CUDAExecutionProvider" in onnxruntime.get_available_providers()
    ):
        provider = "CUDAExecutionProvider"
    else:
        provider = "CPUExecutionProvider"

    # Create inference session
    session = onnxruntime.InferenceSession(
        model_path, sess_options, providers=[provider]
    )

    logger.info(f"ONNX model loaded with provider: {provider}")
    return session


def load_processor(model_dir):
    """Load processor or tokenizer from the model directory."""
    try:
        processor = AutoProcessor.from_pretrained(model_dir)
        logger.info("Loaded AutoProcessor")
        return processor
    except Exception as e:
        logger.warning(f"Failed to load processor: {str(e)}")
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_dir)
            logger.info("Loaded AutoTokenizer")
            return tokenizer
        except Exception as e:
            raise RuntimeError(f"Failed to load processor or tokenizer: {str(e)}")


def process_image(processor, image_path):
    """Process image for ColPali model."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at {image_path}")

    image = Image.open(image_path).convert("RGB")

    # Process image using processor
    try:
        inputs = processor(images=image, return_tensors="np")
    except Exception as e:
        logger.warning(f"Error using processor with standard method: {str(e)}")
        try:
            # Fallback to simply resizing image
            image = image.resize((384, 384))
            # Convert to numpy array in expected format
            image_array = np.array(image).transpose(2, 0, 1)[np.newaxis, ...]
            inputs = {"pixel_values": image_array.astype(np.float32) / 255.0}
        except Exception as e:
            raise RuntimeError(f"Failed to process image: {str(e)}")

    return inputs


def process_query(processor, query):
    """Process query text for ColPali model."""
    try:
        inputs = processor(text=query, return_tensors="np")
    except Exception as e:
        logger.warning(f"Error processing query with standard method: {str(e)}")
        try:
            # Fallback to tokenizer if processor fails
            tokenizer = (
                AutoTokenizer.from_pretrained(processor.name_or_path)
                if hasattr(processor, "name_or_path")
                else None
            )
            if tokenizer:
                inputs = tokenizer(query, return_tensors="np")
            else:
                raise RuntimeError("No tokenizer available")
        except Exception as nested_e:
            raise RuntimeError(f"Failed to process query: {str(nested_e)}")

    return inputs


def get_embeddings(session, inputs):
    """Run inference with ONNX Runtime to get embeddings."""
    # Get input names from model
    input_names = [input.name for input in session.get_inputs()]

    # Prepare inputs in the format expected by the model
    onnx_inputs = {}
    for name in input_names:
        if name in inputs:
            onnx_inputs[name] = inputs[name]

    # Run inference
    outputs = session.run(None, onnx_inputs)

    return outputs


def index_documents(args):
    """Index a directory of document images."""
    if not args.doc_dir:
        raise ValueError(
            "Document directory (--doc_dir) must be specified for indexing"
        )

    doc_dir = Path(args.doc_dir)
    if not doc_dir.exists() or not doc_dir.is_dir():
        raise ValueError(
            f"Document directory {doc_dir} does not exist or is not a directory"
        )

    # Load model and processor
    model_path = os.path.join(args.model_dir, "model.onnx")
    session = load_onnx_model(model_path, args.device)
    processor = load_processor(args.model_dir)

    # Collect document files (images)
    valid_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".gif", ".webp"]
    doc_files = [
        f
        for f in doc_dir.glob("**/*")
        if f.is_file() and f.suffix.lower() in valid_extensions
    ]

    if not doc_files:
        raise ValueError(f"No valid image files found in {doc_dir}")

    logger.info(f"Found {len(doc_files)} document images to index")

    # Process documents and extract embeddings
    document_index = {}
    for doc_file in tqdm(doc_files, desc="Indexing documents"):
        try:
            # Process image
            doc_inputs = process_image(processor, doc_file)

            # Get embeddings
            doc_embeddings = get_embeddings(session, doc_inputs)

            # Store in index
            document_index[str(doc_file)] = {
                "path": str(doc_file),
                "relative_path": str(doc_file.relative_to(doc_dir)),
                "embeddings": doc_embeddings[
                    0
                ],  # Assuming first output is the embedding
            }
        except Exception as e:
            logger.error(f"Error processing document {doc_file}: {str(e)}")

    # Save index
    with open(args.index_path, "wb") as f:
        pickle.dump(document_index, f)

    logger.info(
        f"Successfully indexed {len(document_index)} documents to {args.index_path}"
    )


def calculate_similarity(query_embedding, doc_embedding):
    """Calculate similarity between query and document embeddings."""
    # This is a simplified version of late interaction similarity (ColBERT-style)
    # For each query token, find the maximum similarity with any document token
    similarity = np.einsum("qd,bd->qb", query_embedding, doc_embedding)
    max_sim = np.max(similarity, axis=1)
    return np.sum(max_sim)


def search_documents(args):
    """Search indexed documents with a text query."""
    if not args.query:
        raise ValueError("Query (--query) must be specified for searching")

    if not os.path.exists(args.index_path):
        raise FileNotFoundError(f"Index file not found at {args.index_path}")

    # Load index
    with open(args.index_path, "rb") as f:
        document_index = pickle.load(f)

    logger.info(f"Loaded document index with {len(document_index)} documents")

    # Load model and processor
    model_path = os.path.join(args.model_dir, "model.onnx")
    session = load_onnx_model(model_path, args.device)
    processor = load_processor(args.model_dir)

    # Process query
    query_inputs = process_query(processor, args.query)

    # Get query embeddings
    query_embeddings = get_embeddings(session, query_inputs)

    # Calculate similarity with all documents
    results = []
    for doc_path, doc_data in document_index.items():
        try:
            similarity = calculate_similarity(
                query_embeddings[0], doc_data["embeddings"]
            )
            results.append(
                {
                    "path": doc_data["path"],
                    "relative_path": doc_data["relative_path"],
                    "similarity": float(similarity),
                }
            )
        except Exception as e:
            logger.error(f"Error calculating similarity for {doc_path}: {str(e)}")

    # Sort results by similarity (descending)
    results.sort(key=lambda x: x["similarity"], reverse=True)

    # Return top-k results
    top_results = results[: args.top_k]

    logger.info(f"Search results for query: '{args.query}'")
    for i, result in enumerate(top_results):
        logger.info(
            f"{i+1}. {result['relative_path']} (Score: {result['similarity']:.4f})"
        )

    return top_results


def main():
    args = parse_args()

    if args.mode == "index":
        index_documents(args)
    elif args.mode == "search":
        search_documents(args)
    else:
        raise ValueError(f"Invalid mode: {args.mode}")


if __name__ == "__main__":
    main()
