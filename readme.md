# controlnet_aux_onnx

Fast, self-contained ONNX runtime wrappers and conversion scripts for `controlnet_aux` preprocessors.

All ONNX models share **unified Input/Output specifications** and bake complex pre-processing and post-processing (normalization, padding, flip-augmentation, scribble, safe_step, etc.) directly into the ONNX graph.

- 🤗 **Pre-converted ONNX Models**: [Hugging Face Hub (`Buuta/controlnet_aux_onnx`)](https://huggingface.co/Buuta/controlnet_aux_onnx)

---

## 🌟 Highlights

- **Zero-Config Inference**: The Python wrapper (`src/controlnet_aux_onnx`) automatically downloads required ONNX files from Hugging Face Hub on first run and caches them locally.
- **Unified I/O**: Feed raw `uint8` BGR images (OpenCV format) directly and get `float32` RGB `[0.0 - 1.0]` tensors ready for ControlNet.
- **Fast**: Up to **13x faster on CPU** (tested on Snapdragon X Elite) compared to the original PyTorch implementation.
- **Cross-Platform Ready**: Easily portable to C#, C++, Rust, or Web environments without Python dependencies.

---

## 📁 Repository Structure

```text
controlnet_aux_onnx/
├── src/
│   └── controlnet_aux_onnx/   # Python wrapper matching controlnet_aux API (auto-downloads ONNX)
├── model_converters/          # PyTorch-to-ONNX export scripts for each model & filter
├── models/                    # Directory where ONNX models are exported
├── tests/                     # pytest validation & benchmark suites
├── pyproject.toml
├── LICENSE
└── README.md
```

---

## 🚀 Quick Start (Python)

### 1. Installation

```bash
git clone https://github.com/buuta-buta-butaata/controlnet_aux_onnx
cd controlnet_aux_onnx

# Install core runtime dependencies
pip install onnxruntime numpy huggingface_hub

# Install controlnet_aux_onnx
pip install -e .
```

*(Note: If you want to run the conversion scripts in `model_converters/` or `tests/`, please also install `torch`, `transformers`, `controlnet_aux`, `opencv-python`, and `pytest`.)*

### 2. Usage Example

The wrapper mirrors the behavior of `controlnet_aux`. Missing ONNX models are automatically fetched from Hugging Face into the specified directory.
You can directly specify ONNX Runtime options(`providers`, `provider_options`, and `sess_options`) when initializing each detector.

[Python sample script](example.py)
<!-- [C# sample script](example.cs) -->

---

## 🛠️ Exporting Models Locally (`model_converters/`)

If you want to export the ONNX models yourself:

```bash
# Example: Export HED and post-processing scribble filter
python model_converters/export_hed.py
python model_converters/export_filter_scribble.py
```

### ⚠️ Note on LeReS (`export_leres.py`)
Due to the original **Adobe Research License** (which prohibits redistribution of the research materials/weights), `leres.onnx` is **not hosted on Hugging Face**. 
To use LeReS, please generate the ONNX file locally by running:
```bash
python model_converters/export_leres.py
```

---

## 🧪 Test Design & Validation Criteria

Because post-processing filters can generate binary (`[0, 255]`) or ternary (`[0, 127, 255]`) images, the test validation criteria are designed as follows to account for quantization boundary shifts:

1. **Exact Match Rate**: More than **95%** of all pixels must match identically.
2. **Quantization Tolerance**: Any remaining discrepancies must fall within **1 quantization step** (e.g., `1` for standard `[0-255]` RGB images, or `128` for ternary images).

*(Note: For base models without post-processing filters, over **97%** of pixels match identically, achieving a **100%** match rate within a `1/255` tolerance.)*

## 🧪 Running Tests & Benchmarks (`tests/`)

All ONNX models are verified against the original `controlnet_aux` outputs using `pytest`.

```bash
# Run accuracy tests and speed benchmarks
pytest tests/ -v -s

# Run for a specific model only
pytest tests/ -v -s -k "hed"
```

---

## 📜 License

- **Code & Conversion Scripts**: Licensed under the **Apache License 2.0**.
- **Model Weights**: Each exported ONNX model inherits the license of its original pre-trained weights (Apache 2.0, MIT, or Adobe Research License). Please refer to the [Hugging Face Model Card](https://huggingface.co/Buuta/controlnet_aux_onnx) for the full license matrix.


