import argparse
from dotenv import load_dotenv

from lmm_vlm.common import ModelResult, timed_call, validate_image
from lmm_vlm.providers import (
    DEFAULT_GEMINI_MODEL,
    DEFAULT_LLAMA_MODEL,
    ask_gemini,
    ask_llama,
    ask_openai,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Ask a vision-language model a question about an image."
    )
    parser.add_argument(
        "--provider",
        required=True,
        choices=("openai", "gemini", "llama"),
        help="Model provider to use.",
    )
    parser.add_argument("--image", required=True, help="Path to a PNG/JPG image.")
    parser.add_argument("--question", required=True, help="Question about the image.")
    parser.add_argument("--model", help="Optional model identifier override.")
    return parser.parse_args()


def main():
    load_dotenv()
    args = parse_args()

    if not args.question.strip():
        raise ValueError("Question cannot be empty.")

    image_path, mime_type = validate_image(args.image)

    if args.provider == "openai":
        model = args.model or "gpt-4o"
        call = lambda: ask_openai(image_path, mime_type, args.question, model)
    elif args.provider == "gemini":
        model = args.model or DEFAULT_GEMINI_MODEL
        call = lambda: ask_gemini(image_path, mime_type, args.question, model)
    else:
        model = args.model or DEFAULT_LLAMA_MODEL
        call = lambda: ask_llama(image_path, args.question, model)

    answer, elapsed = timed_call(call)
    result = ModelResult(args.provider, model, answer, elapsed)

    print(f"\nProvider: {result.provider}")
    print(f"Model: {result.model}")
    print(f"Elapsed: {result.elapsed_seconds:.2f} seconds")
    print("\nAnswer:\n")
    print(result.answer)


if __name__ == "__main__":
    main()
