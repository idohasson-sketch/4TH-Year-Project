import os
from onnxruntime.quantization import quantize_dynamic, QuantType

# --- CONFIGURATION ---
SOURCE_PATH = os.path.join(os.path.expanduser("~"), 'Downloads', 'model.onnx')
PROJECT_DIR = "/Users/idohasson/4th year project"
QUANTIZED_PATH = os.path.join(PROJECT_DIR, 'model_quantized.onnx')

if not os.path.exists(SOURCE_PATH):
    raise FileNotFoundError(f"Model not found at: {SOURCE_PATH}")

print(f"[*] Found model at: {SOURCE_PATH}")

# Perform INT8 Quantization
print(f"[+] Quantizing to INT8...")
quantize_dynamic(
    model_input=SOURCE_PATH,
    model_output=QUANTIZED_PATH,
    weight_type=QuantType.QUInt8
)

final_size = os.path.getsize(QUANTIZED_PATH) / 1024 / 1024
print(f"\n[V] Success! Quantized model saved to: {QUANTIZED_PATH}")
print(f"[V] Final Model Size: {final_size:.2f} MB")
