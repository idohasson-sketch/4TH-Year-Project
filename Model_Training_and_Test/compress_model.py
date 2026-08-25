"""
===============================================================================
A-EYE TRACKER — INT8 Dynamic Quantization Pipeline
===============================================================================
Purpose:
Quantizes trained Float32 ONNX models into INT8 representation via ONNX Runtime,
slashing binary footprint for embedded OpenMV deployment.
===============================================================================
"""

import os
import argparse
from onnxruntime.quantization import quantize_dynamic, QuantType


def quantize_onnx_model(input_path: str, output_path: str):
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"[X] Source ONNX model not found at: {input_path}")

    output_dir = os.path.dirname(os.path.abspath(output_path))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    print(f"[*] Quantizing Float32 ONNX -> INT8: {input_path}")
    quantize_dynamic(
        model_input=input_path,
        model_output=output_path,
        weight_type=QuantType.QUInt8
    )

    original_size = os.path.getsize(input_path) / (1024 * 1024)
    quantized_size = os.path.getsize(output_path) / (1024 * 1024)
    reduction = ((original_size - quantized_size) / original_size) * 100

    print(f"[V] Quantization Complete -> {output_path}")
    print(f"    Original: {original_size:.2f} MB | Quantized: {quantized_size:.2f} MB (-{reduction:.1f}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A-EYE Tracker ONNX INT8 Quantizer")
    parser.add_argument("--input_onnx", type=str, required=True, help="Path to Float32 .onnx model")
    parser.add_argument("--output_quantized_onnx", type=str, required=True, help="Path to save INT8 .onnx model")
    args = parser.parse_args()

    quantize_onnx_model(args.input_onnx, args.output_quantized_onnx)
