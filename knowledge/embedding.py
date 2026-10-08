import json
import os
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types


MODEL = "gemini-embedding-001"


def embed_file(filename):
    folder = Path(__file__).resolve().parent
    input_path = folder / filename
    output_path = input_path.with_name(
        f"{input_path.stem}_embeddings.json"
    )

    if not input_path.is_file():
        raise FileNotFoundError(f"فایل پیدا نشد: {input_path}")

    # خواندن کلید از فایل .env کنار کد
    load_dotenv(folder / ".env")
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key or not api_key.strip():
        raise ValueError("کلید GEMINI_API_KEY تنظیم نشده است.")

    chunks = json.loads(
        input_path.read_text(encoding="utf-8-sig")
    )

    if not isinstance(chunks, list) or not chunks:
        raise ValueError("فایل ورودی باید لیستی از قطعه‌ها باشد.")

    # بررسی تمام قطعه‌ها قبل از ارسال درخواست
    for index, chunk in enumerate(chunks, start=1):
        if not isinstance(chunk, dict):
            raise ValueError(f"ساختار قطعهٔ {index} نامعتبر است.")

        text = chunk.get("text")

        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"متن قطعهٔ {index} خالی یا نامعتبر است.")

    embedded_chunks = []

    with genai.Client(api_key=api_key.strip()) as client:
        for chunk in chunks:
            response = client.models.embed_content(
                model=MODEL,
                contents=chunk["text"],
                config=types.EmbedContentConfig(
                    task_type="RETRIEVAL_DOCUMENT",
                ),
            )

            if not response.embeddings:
                raise RuntimeError("API برداری برنگرداند.")

            vector = response.embeddings[0].values

            if not vector:
                raise RuntimeError("بردار دریافتی خالی است.")

            embedded_chunks.append({
                **chunk,
                "embedding": vector,
                "embedding_model": MODEL,
            })

    # ذخیره فقط پس از موفقیت تمام درخواست‌ها
    output_path.write_text(
        json.dumps(
            embedded_chunks,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    return output_path


if __name__ == "__main__":
    embed_file(
        "synthetic_medibridge_healthcare_rag_dataset_clean_chunks.json"
    )