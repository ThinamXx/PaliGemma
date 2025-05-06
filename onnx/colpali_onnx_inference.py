"""
ColPali ONNX Inference

This script demonstrates how to use the converted ColPali ONNX model for inference.
It shows how to load the ONNX model, process input images, and get document embeddings.

Usage:
    python colpali_onnx_inference.py --model_dir colpali_onnx --image_path document.jpg

"""

import argparse
import os
import numpy as np
import torch
import onnxruntime
from PIL import Image
from transformers import AutoProcessor, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="ColPali ONNX Inference")
    parser.add_argument(
        "--model_dir",
        type=str,
        default="colpali_onnx",
        help="Directory containing ONNX model and processor/tokenizer",
    )
    parser.add_argument(
        "--image_path",
        type=str,
        required=True,
        help="Path to document image for embedding",
    )
    parser.add_argument(
        "--query",
        type=str,
        default="",
        help="Optional query text to calculate similarity with document",
    )
    return parser.parse_args()


def load_onnx_model(model_path):
    """Load ONNX model with ONNX Runtime."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"ONNX model not found at {model_path}")

    # Set up ONNX Runtime session
    sess_options = onnxruntime.SessionOptions()
    sess_options.graph_optimization_level = (
        onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    )

    # Create inference session
    session = onnxruntime.InferenceSession(
        model_path,
        sess_options,
        providers=["CPUExecutionProvider"],  # Use CUDAExecutionProvider for GPU
    )

    return session


def process_image(processor, image_path):
    """Process image for ColPali model."""
    # Load image
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at {image_path}")

    image = Image.open(image_path).convert("RGB")

    # Process image using processor
    inputs = processor(images=image, return_tensors="np")

    return inputs


def process_query(processor, query):
    """Process query text for ColPali model."""
    # Process query using processor
    inputs = processor(text=query, return_tensors="np")

    return inputs


def get_embeddings(session, inputs):
    """Run inference with ONNX Runtime to get embeddings."""
    # Get input names from model
    input_names = [input.name for input in session.get_inputs()]

    # Prepare inputs in the format expected by the model
    onnx_inputs = {name: inputs[name] for name in input_names if name in inputs}

    # Run inference
    outputs = session.run(None, onnx_inputs)

    return outputs


def main():
    args = parse_args()

    # Paths
    model_path = os.path.join(args.model_dir, "model.onnx")

    # Load processor
    try:
        processor = AutoProcessor.from_pretrained(args.model_dir)
    except:
        try:
            processor = AutoTokenizer.from_pretrained(args.model_dir)
        except Exception as e:
            raise RuntimeError(f"Failed to load processor/tokenizer: {str(e)}")

    # Load ONNX model
    print(f"Loading ONNX model from {model_path}...")
    session = load_onnx_model(model_path)

    # Process image
    print(f"Processing image from {args.image_path}...")
    image_inputs = process_image(processor, args.image_path)

    # Get image embeddings
    print("Getting document embeddings...")
    doc_embeddings = get_embeddings(session, image_inputs)

    print(f"Document embedding shape: {doc_embeddings[0].shape}")

    # If query provided, calculate similarity
    if args.query:
        print(f"Processing query: '{args.query}'")
        query_inputs = process_query(processor, args.query)
        query_embeddings = get_embeddings(session, query_inputs)

        print(f"Query embedding shape: {query_embeddings[0].shape}")

        # Calculate similarity if both embeddings are available
        if doc_embeddings and query_embeddings:
            print("Calculating late interaction similarity score...")
            # Use late interaction similarity (max of dot products between query and document tokens)
            # This is a simplified version of the ColBERT scoring
            doc_emb = doc_embeddings[0]
            query_emb = query_embeddings[0]

            # Late interaction scoring (simplified version)
            # For each query token, find the maximum similarity with any document token
            similarity_scores = (
                np.einsum("bnd,csd->bcns", query_emb, doc_emb).max(axis=3).sum(axis=2)
            )
            print(f"Similarity score: {similarity_scores}")


if __name__ == "__main__":
    main()
