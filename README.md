# Large Multimodal Models (LMMs) / Vision-Language Models

A small Python command-line project for asking a question about an image using:
- OpenAI GPT-4o (hosted API)
- Google Gemini (hosted API)
- Meta Llama 3.2 Vision (local inference)

The same image and question are passed to the selected model. The program prints the answer and elapsed time. This is an implementation framework; it does not include benchmark results.

## 1. Requirements

- Python 3.10 or newer
- An image in PNG, JPG or JPEG format
- For hosted models: a valid API key and internet access
- For local Llama: Hugging Face access approval for the model, sufficient RAM/VRAM, and a compatible PyTorch installation

## 2. Setup

```bash
# Clone your GitHub repository
git clone https://github.com/shrijit007/lmm-vision-language-models.git
cd lmm-vision-language-models

# Create and activate a virtual environment
python -m venv .venv

# macOS / Linux
source .venv/bin/activate

# Windows PowerShell
# .venv\Scripts\Activate.ps1

# Install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt

# Create your local environment file
cp .env.example .env
```

On Windows PowerShell, use `Copy-Item .env.example .env` instead of `cp`.

## 3. Configure API keys

Edit `.env` and add the keys for the hosted provider(s) you intend to use:

```env
OPENAI_API_KEY=your_openai_api_key
GEMINI_API_KEY=your_google_gemini_api_key
```

Keep `.env` private. It is excluded by `.gitignore`; never commit API keys.

## 4. Run the project

Place an image in the project folder (for example, `photo.jpg`).

### OpenAI GPT-4o

```bash
python main.py --provider openai --image photo.jpg --question "Describe this image in two sentences."
```

### Google Gemini

```bash
python main.py --provider gemini --image photo.jpg --question "What is the main object in this image?"
```

The default Gemini model is `gemini-2.5-flash`. Override it with `--model` if needed:

```bash
python main.py --provider gemini --model gemini-2.5-flash --image photo.jpg --question "Read the total on this receipt."
```

### Local Llama 3.2 Vision

Install the additional local inference dependencies (PyTorch can be a large download):

```bash
pip install -r requirements-local.txt
```

First, accept Meta's applicable model terms on Hugging Face and authenticate:

```bash
huggingface-cli login
```

Then run (the 11B vision model requires substantial memory and may need a capable GPU):

```bash
python main.py --provider llama --image photo.jpg --question "Describe this image."
```

The default model ID is `meta-llama/Llama-3.2-11B-Vision-Instruct`. You can override it with `--model`.

## 5. Project structure

```text
lmm-vision-language-models/
├── lmm_vlm/
│   ├── __init__.py
│   ├── common.py
│   └── providers.py
├── main.py
├── requirements.txt
├── requirements-local.txt
├── .env.example
├── .gitignore
└── README.md
```

- `common.py`: image validation, MIME detection and timing/result structure.
- `providers.py`: provider-specific inference adapters.
- `main.py`: command-line interface and result display.

## 6. Example output

```text
Provider: openai
Model: gpt-4o
Elapsed: 2.41 seconds

Answer:
The image shows ...
```

Actual output and timing depend on the model version, network, hardware, image and prompt.

## 7. Evaluation notes

For a fair comparison, use the same image, question and equivalent generation settings for every model. Record:
- exact model identifier and test date
- image dimensions and any preprocessing
- answer correctness against a prepared reference
- unsupported claims / hallucinations
- elapsed time and hosted API cost, where applicable

This repository does not claim measured accuracy, latency or cost. Those values must be collected by running a defined test set.

## 8. Troubleshooting

- **Missing API key:** check `.env` and ensure the key name matches the provider.
- **Model not found / unavailable:** verify the current model identifier in the provider's documentation.
- **Quota or authentication error:** check account access, billing/quota and credentials.
- **Llama access denied:** accept the model license and ensure your Hugging Face account has access.
- **Out of memory:** local Llama 3.2 Vision 11B is resource-intensive; use supported hardware or run a hosted provider.
- **Image format error:** use a valid PNG, JPG or JPEG image.

## 9. Privacy

Hosted providers receive the image and question. Do not upload private photographs, identity documents or financial records unless the transfer is appropriate and the provider's data policy has been reviewed. Local inference keeps image processing on your machine, subject to your local environment and model setup.

## 10. License and attribution

This project is an educational implementation framework. Provider SDKs and model weights remain subject to their respective licenses and terms. Meta Llama 3.2 Vision weights are not redistributed in this repository.
