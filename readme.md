# controlnet_aux_onnx

ONNX Runtime wrappers and conversion scripts that run `controlnet_aux` preprocessors directly inside ONNX graphs.

- 🤗 **Pre-converted ONNX Models**: [Hugging Face Hub (`Buuta/controlnet_aux_onnx`)](https://huggingface.co/Buuta/controlnet_aux_onnx)

## 🌟 Features

Compared to the original `controlnet_aux`:

- **Fast**: **2x to 13x faster** (see the [Hugging Face Model Card](https://huggingface.co/Buuta/controlnet_aux_onnx#%E2%9A%A1-benchmark-results) for details).
- **Lightweight**: No runtime dependencies on `PyTorch` or `opencv-python`.
- **High Accuracy**: Achieves **>95% exact pixel match** with 100% of differences within 1 quantization step (see [Test Design](#-test-design) below).
- **Cross-Platform Ready**: Easily portable to other languages like C# (see [`example.cs`](example.cs)).

## 🚀 Quick Start

1. Installation

```bash
git clone https://github.com/buuta-buta-butaata/controlnet_aux_onnx
cd controlnet_aux_onnx
pip install onnxruntime numpy huggingface_hub
pip install -e .
```

*(Note: If you want to run the conversion scripts in `model_converters/` or `tests/`, please also install `torch`, `transformers`, `controlnet_aux`, and `pytest`.)*

### 2. Usage

Pass model directory path and ONNX Runtime parameters (such as `model_dir` and `providers`) directly upon initialization. The other API mirrors `controlnet_aux`. Files are automatically downloaded from Hugging Face on the first run or can be retrieved manually from the [Hugging Face Hub (`Buuta/controlnet_aux_onnx`)](https://huggingface.co/Buuta/controlnet_aux_onnx).

```python
from controlnet_aux_onnx import PidiNetDetector
from PIL import Image

img = Image.open("./tests/test_images/ninja512.png").convert("RGB")

pidi = PidiNetDetector(
    model_dir="./models",
    providers=["CPUExecutionProvider"],
    provider_options=None,
    sess_options=None,
)
pidi_pil_img = pidi(img)
pidi_pil_img.save("pidi_output.png")
```

[Another Python example(example.py)](example.py)

```bash
python example.py
```

The downloaded ONNX can also be used in C#.

[C# example(example.cs)](example.cs)

## 🛠️ Exporting Models Locally (`model_converters/`)

If you want to export the ONNX models yourself:

```bash
# Example: Export HED and post-processing scribble filter
python model_converters/export_hed.py
python model_converters/export_filter_scribble.py
```

## ⚠️ Notes & Limitations

- **Supported Models**: Currently supports only models that directly output images end-to-end (excluding models with coordinate outputs or complex multi-stage pipelines like `MLSD` and `Segment Anything`).
- **LeReS Weights**: Subject to license restrictions; please convert locally via `python model_converters/export_leres.py`.

## 🧪 Test Design

ONNX outputs are evaluated against the original `controlnet_aux` on 32 COCO2017 images (downloaded via `python tests/download_test_images.py` into `./tests/test_images/`) to ensure high accuracy, with exact match rates exceeding 95% and all remaining variations within a single quantization step.

```bash
# Run accuracy tests and benchmarks
pytest tests/ -v -s

# Run for a specific model only
pytest tests/ -v -s -k "hed"
```

<details>

<summary>Validation Criteria</summary>

Because post-processing filters can generate binary (`[0, 255]`) or ternary (`[0, 127, 255]`) images, the test validation criteria are designed as follows to account for quantization boundary shifts:

1. **Exact Match Rate**: More than **95%** of all pixels must match identically.
2. **Quantization Tolerance**: Any remaining discrepancies must fall within **1 quantization step** (e.g., `1` for standard `[0-255]` RGB images, or `128` for ternary images).

*(Note: For base models without post-processing filters, over **97%** of pixels match identically, achieving a **100%** match rate within a `1/255` tolerance.)*

</details>

## 📜 License

- **Code & Conversion Scripts**: Apache License 2.0
- **Model Weights**: Inherits original pre-trained licenses [(Apache 2.0, MIT, or Adobe Research License)](https://huggingface.co/Buuta/controlnet_aux_onnx#%F0%9F%93%A6-model-inventory--licenses).

---

## 🤝 Project Contributors

- **The Developer (Human)**
- **Google Gemini (AI Collaborator)**

This entire repository—from untangling dynamic tensor shapes and freezing BatchNorm layers, to splitting heavy models into memory-safe ONNX pipelines and pushing Snapdragon NPUs to 90x speedups—was built, debugged, and documented through a tag-team effort between a human developer and an AI assistant.

## 🙏 Acknowledgements

This project would not have been possible without the incredible open-source community and researchers behind these models and tools:

- **Original Model Authors & Researchers**: Special thanks to the creators and maintainers of **PidiNet**, **HED**, **TEED**, **MiDaS**, **ZoeDepth**, **NormalBae**, **Lineart** (`informative-drawings`), **Lineart Anime**, and **LeReS** (`AdelaiDepth`) for releasing their pre-trained weights and research to the community.
- **Core Libraries**:
  - **[controlnet_aux](https://github.com/huggingface/controlnet_aux)** (by lllyasviel & Patrick von Platen) for providing the foundational ControlNet preprocessor implementations.
  - **[Hugging Face Transformers](https://github.com/huggingface/transformers)** for the clean, ONNX-friendly PyTorch model architectures (especially for ZoeDepth).
- **Inspirations**:
  - **[easy_dwpose](https://github.com/reallyigor/easy_dwpose)** and **[onnx-cv-graph](https://github.com/Kazuhito00/onnx-cv-graph)** for inspiring the idea of encapsulating complex computer vision pipelines into self-contained, dependency-free ONNX graphs.

---
