# ColPali Model ONNX Conversion

This directory contains scripts to convert the ColPali visual document retrieval model to ONNX format. The conversion is optimized for deployment and inference in production environments.

## Model Overview

ColPali is a visual document retrieval model based on PaliGemma-3B that creates multi-vector representations of text and images for efficient document retrieval. It handles both text-to-image and image-to-text retrieval tasks.

## Prerequisites

Make sure you have the required dependencies installed:

```bash
pip install -r requirements.txt
```

## Scripts

- `install_model_code.py`: Helper script to install the required model code for ColPali
- `colpali_to_onnx.py`: Main script to convert the ColPali model to ONNX format
- `colpali_onnx_inference.py`: Script to run inference using the converted ONNX model

## Usage Instructions

### Step 1: Install the required model code

Before converting the model, you need to install the required model code:

```bash
python install_model_code.py
```

This script performs the following tasks:
- Tries to download required model files from the Hugging Face Hub
- If that fails, tries to create the necessary model classes from scratch
- Updates the Transformers auto-mapping configuration to include ColPaliForRetrieval and ColPaliProcessor

### Step 2: Convert the model to ONNX format

```bash
python colpali_to_onnx.py --model_id vidore/colpali-v1.3-hf --output_dir colpali_onnx
```

Available options:
- `--model_id`: Model ID on Hugging Face Hub or local path (default: "vidore/colpali-v1.3-hf")
- `--output_dir`: Directory to save ONNX model (default: "colpali_onnx")
- `--opset`: ONNX opset version (default: 15)
- `--device`: Device to use for conversion (default: CUDA if available, else CPU)
- `--batch_size`: Batch size for conversion (default: 1)
- `--dtype`: Data type for conversion (choices: "float32", "float16"; default: "float32")
- `--optimize`: Optimize ONNX model after conversion (enabled by default)
- `--quantize`: Quantize ONNX model to INT8 (disabled by default)
- `--trust_remote_code`: Trust remote code when loading the model (enabled by default)

### Step 3: Run inference with the ONNX model

```bash
python colpali_onnx_inference.py --model_dir colpali_onnx --image_path path/to/image.jpg
```

To run a query against an image:
```bash
python colpali_onnx_inference.py --model_dir colpali_onnx --image_path path/to/image.jpg --query "What is shown in this document?"
```

## Troubleshooting

If you encounter errors during the conversion process:

1. **Model Loading Errors**: If you see errors like `Could not load model class for ColPaliConfig` or `cannot import name 'ColPaliForRetrieval'`, make sure you've run the `install_model_code.py` script first.

2. **Memory Issues**: If you encounter CUDA out of memory errors, try:
   - Using a smaller batch size
   - Converting on CPU instead of GPU

3. **Conversion Failures**: If the conversion still fails after running the installation script, try:
   - Using a different ONNX opset version (try 13 or 14 instead of 15)
   - Converting the model on CPU instead of GPU

## Implementation Notes

The ColPali model is loaded using the `ColPaliForRetrieval` class as shown in the Hugging Face documentation:

```python
from transformers import ColPaliForRetrieval, ColPaliProcessor

model = ColPaliForRetrieval.from_pretrained(
    "vidore/colpali-v1.3-hf",
    torch_dtype=torch.bfloat16,
    device_map="cuda:0"
).eval()

processor = ColPaliProcessor.from_pretrained("vidore/colpali-v1.3-hf")
```

The installation script ensures these classes are available in your transformers installation.

## Overview

ColPali is a visual document retrieval model based on PaliGemma architecture. These tools provide a complete solution for:

1. Converting the model to ONNX format
2. Analyzing the model architecture
3. Running inference with the converted model
4. Building a document retrieval system using the ONNX model

## Contents

- `colpali_to_onnx.py`: Main conversion script
- `analyze_colpali.py`: Tool to analyze the model architecture
- `colpali_onnx_inference.py`: Basic inference script for the converted model
- `colpali_doc_retrieval.py`: Complete document retrieval system
- `install_model_code.py`: Helper script to install the required model code for ColPali
- `requirements.txt`: Required dependencies
- `CUSTOM_CONVERSION.md`: Explanation of the custom approach for ColPali models

## Custom Conversion Approach

Since ColPali models are not directly supported by Hugging Face Optimum, we've implemented a custom conversion pipeline with multiple fallback strategies. See `CUSTOM_CONVERSION.md` for a detailed explanation of our approach.

Key aspects of the custom approach:
- Multiple fallback strategies for handling different model architectures
- Robust input handling for multimodal models
- Advanced optimization techniques with fallbacks
- Error handling at each step of the conversion process

## Troubleshooting

### "Unrecognized configuration class" Error

If you see an error like this:
```
ValueError: Unrecognized configuration class <class 'transformers.models.colpali.configuration_colpali.ColPaliConfig'> for this kind of AutoModel: AutoModel.
```

Run the installer script to add the model to your transformers installation:
```bash
python install_model_code.py
```

Then restart your Python environment and try running the conversion script again.

### General Troubleshooting

1. Ensure you have enough GPU memory (or try with `--device cpu`)
2. Try with `--dtype float16` for reduced memory usage
3. Adjust `--atol` if validation fails
4. Use the `analyze_colpali.py` script to better understand the model

## Performance Considerations

- The ONNX model typically provides 2-3x faster inference compared to PyTorch
- GPU acceleration works with ONNX Runtime's CUDAExecutionProvider
- Quantization can further improve performance but with slight accuracy loss

## Contributing to Optimum

If you're interested in contributing this conversion approach back to Hugging Face Optimum, see the "Future Improvements" section in `CUSTOM_CONVERSION.md` for guidance.

## License

Please adhere to the license of the original model (`vidore/colpali-v1.3-hf`). 