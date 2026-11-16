import base64
import os
from pathlib import Path

from google import genai
from google.genai import types
from openai import OpenAI
from PIL import Image


DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
DEFAULT_LLAMA_MODEL = "meta-llama/Llama-3.2-11B-Vision-Instruct"


def ask_openai(image_path: Path, mime_type: str, question: str, model: str) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY is missing. Add it to your .env file.")

    client = OpenAI(api_key=api_key)
    encoded = base64.b64encode(image_path.read_bytes()).decode("utf-8")
    data_url = f"data:{mime_type};base64,{encoded}"

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }
        ],
        max_tokens=300,
        temperature=0,
    )
    answer = response.choices[0].message.content
    return answer or ""


def ask_gemini(image_path: Path, mime_type: str, question: str, model: str) -> str:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError("GEMINI_API_KEY is missing. Add it to your .env file.")

    client = genai.Client(api_key=api_key)
    image_bytes = image_path.read_bytes()
    response = client.models.generate_content(
        model=model,
        contents=[
            types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
            question,
        ],
        config=types.GenerateContentConfig(temperature=0),
    )
    return response.text or ""


def ask_llama(image_path: Path, question: str, model_id: str) -> str:
    # Import local-inference dependencies only when this provider is selected.
    import torch
    from transformers import AutoProcessor, MllamaForConditionalGeneration

    # The Llama 3.2 Vision checkpoint is gated; Hugging Face access and license
    # acceptance are required before this model can be downloaded.
    model = MllamaForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(model_id)

    image = Image.open(image_path).convert("RGB")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": question},
            ],
        }
    ]
    prompt = processor.apply_chat_template(
        messages, add_generation_prompt=True
    )
    inputs = processor(
        image,
        prompt,
        add_special_tokens=False,
        return_tensors="pt",
    ).to(model.device)

    output = model.generate(**inputs, max_new_tokens=200)
    prompt_length = inputs["input_ids"].shape[-1]
    generated_tokens = output[0][prompt_length:]
    return processor.decode(generated_tokens, skip_special_tokens=True).strip()
