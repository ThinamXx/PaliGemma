# Custom ONNX Conversion for ColPali Models

This document explains the custom approach taken to convert the ColPali multimodal document understanding models to ONNX format, since they are not directly supported by Hugging Face Optimum.

## The Challenge

The ColPali models (`vidore/colpali-v1.3-hf` and `vidore/colqwen2-v1.0-merged`) are based on the PaliGemma architecture, which has specific components that make standard conversion challenging:

1. Multimodal architecture combining vision and text processing
2. Custom attention mechanisms not fully supported by standard ONNX exporters
3. Complex tokenization and image processing pipelines

## Our Approach

We've implemented a robust conversion pipeline with multiple fallback strategies:

### 1. First Attempt: Optimum Export

We first try to use Hugging Face Optimum's `main_export` function with the "feature-extraction" task. This may work for some model variants but often fails with custom architectures.

### 2. Fallback: Manual PyTorch ONNX Export

When Optimum fails, we use PyTorch's native ONNX export with a carefully constructed input pipeline:

- Analyze the model's processor to understand input requirements
- Create appropriate dummy inputs based on processor capabilities
- Handle both image and text input paths
- Set up dynamic axes for variable batch sizes and sequence lengths

### 3. Resilient Export Process

The export process includes multiple fallback mechanisms:

- If complex inputs fail, try with simpler inputs
- If processor-based inputs fail, fall back to standard dummy tensors
- Catch and handle exceptions at each step

### 4. Post-Export Optimization

After successful export, we apply:

1. ONNX model validation and checking
2. Optimization using onnxruntime.transformers (with fallback to basic optimization)
3. Optional quantization to INT8 for reduced model size

## Implementation Details

### Input Handling

The script handles inputs in layers of fallback:

1. First, try to use the model's processor with both image and text inputs
2. If that fails, try with separate image processor and tokenizer if available
3. Fall back to standard random tensors matching expected shapes
4. As a last resort, try with only image inputs

### Optimization Strategy

We use a two-tier optimization approach:

1. First try advanced optimization with onnxruntime.transformers
2. If that fails, use basic graph optimizations from onnxruntime

## Extension for Other Models

This approach can be extended to other multimodal models not directly supported by Optimum:

1. Analyze the model's architecture and input requirements
2. Create appropriate dummy inputs based on this analysis
3. Implement fallback strategies for various failure points
4. Apply post-export optimizations as appropriate

## Future Improvements

Ideally, this custom approach would be contributed back to Hugging Face Optimum as a proper integration:

1. Create custom OnnxConfig implementation for PaliGemma-based models
2. Add model mapping in Optimum's model configurations
3. Implement proper patching for attention mechanisms and other custom components 

```bash
python colpali_to_onnx.py --model_id vidore/colpali-v1.3-hf --output_dir ../colpali_onnx --dtype float16
```

```bash
python colpali_to_onnx.py --model_id vidore/colpali-v1.3-hf --output_dir ../colpali_onnx --dtype float16 --optimize
```
