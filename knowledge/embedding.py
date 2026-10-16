import json
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


MODEL = "text-embedding-3-small"
BASE_URL = "https://api.gapgpt.app/v1"


def embed_file(filename):
    folder = Path(__file__).resolve().parent
    input_path = folder / filename
    output_path = input_path.with_name(
        f"{input_path.stem}_embeddings.json"
    )

    if not input_path.is_file():
        raise FileNotFoundError(f"فایل پیدا نشد: {input_path}")

    load_dotenv(folder / ".env")
    api_key = os.getenv("GAPGPT_API_KEY", "").strip()

    if not api_key:
        raise ValueError("کلید GAPGPT_API_KEY تنظیم نشده است.")

    chunks = json.loads(
        input_path.read_text(encoding="utf-8-sig")
    )

    if not isinstance(chunks, list) or not chunks:
        raise ValueError("فایل ورودی باید لیستی از قطعه‌ها باشد.")

    for index, chunk in enumerate(chunks, start=1):
        if not isinstance(chunk, dict):
            raise ValueError(f"ساختار قطعهٔ {index} نامعتبر است.")

        text = chunk.get("text")

        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"متن قطعهٔ {index} خالی یا نامعتبر است.")

    embedded_chunks = []
    dimensions = None

    with OpenAI(
        base_url=BASE_URL,
        api_key=api_key,
        timeout=30.0,
        max_retries=2,
    ) as client:
        for chunk in chunks:
            response = client.embeddings.create(
                model=MODEL,
                input=chunk["text"],
            )

            if not response.data:
                raise RuntimeError("API برداری برنگرداند.")

            vector = response.data[0].embedding

            if not vector:
                raise RuntimeError("بردار دریافتی خالی است.")

            if dimensions is None:
                dimensions = len(vector)
            elif len(vector) != dimensions:
                raise RuntimeError("ابعاد بردارها یکسان نیست.")

            embedded_chunks.append({
                **chunk,
                "embedding": vector,
                "embedding_model": MODEL,
                "embedding_provider": "gapgpt",
            })

    # خروجی قبلی فقط پس از موفقیت تمام درخواست‌ها جایگزین می‌شود
    output_path.write_text(
        json.dumps(
            embedded_chunks,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )

    return output_path


if __name__ == "__main__":
    embed_file(
        "synthetic_medibridge_healthcare_rag_dataset_clean_chunks.json"
    )