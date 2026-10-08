import json
import re
from pathlib import Path


def split_long_text(text, max_words=250, overlap=40):
    """تقسیم متن بلند با هم‌پوشانی برای حفظ ارتباط قطعه‌ها."""
    words = text.split()

    if len(words) <= max_words:
        return [text]

    parts = []
    start = 0

    while start < len(words):
        end = min(start + max_words, len(words))
        parts.append(" ".join(words[start:end]))

        if end == len(words):
            break

        start = end - overlap

    return parts


def chunk_text(text, source, max_words=250, overlap=40):
    if max_words <= 0:
        raise ValueError("max_words باید مثبت باشد.")

    if not 0 <= overlap < max_words:
        raise ValueError("overlap باید بین صفر و max_words باشد.")

    # خروجی Cleaning بین رکوردها یک خط خالی دارد
    blocks = re.split(r"\n\s*\n", text.strip())

    chunks = []
    section = ""

    for block in blocks:
        block = block.strip()

        if not block:
            continue

        # عنوان بخش را به‌عنوان metadata نگه می‌داریم
        if re.fullmatch(r"\d{2}\s+[A-Z][A-Z0-9 /&_-]*", block):
            section = block
            continue

        # شناسهٔ رکورد، اگر موجود باشد
        match = re.search(
            r"^(ENTITY_ID|FACILITY_ID|POLICY_ID|MAPPING_ID):\s*(\S+)",
            block,
            flags=re.MULTILINE,
        )

        record_id = match.group(2) if match else None

        # هم‌پوشانی فقط داخل همان رکورد انجام می‌شود
        parts = split_long_text(block, max_words, overlap)

        for part_number, part in enumerate(parts, start=1):
            chunks.append({
                "chunk_id": f"chunk_{len(chunks) + 1:04d}",
                "text": part,
                "metadata": {
                    "source": source,
                    "section": section,
                    "record_id": record_id,
                    "part": part_number,
                    "total_parts": len(parts),
                    "word_count": len(part.split()),
                },
            })

    return chunks


def chunk_file(filename, max_words=250, overlap=40):
    folder = Path(__file__).resolve().parent
    input_path = folder / filename
    output_path = input_path.with_name(
        f"{input_path.stem}_chunks.json"
    )

    if not input_path.is_file():
        raise FileNotFoundError(f"فایل پیدا نشد: {input_path}")

    text = input_path.read_text(encoding="utf-8-sig")

    chunks = chunk_text(
        text=text,
        source=input_path.name,
        max_words=max_words,
        overlap=overlap,
    )

    if not chunks:
        raise ValueError("متنی برای قطعه‌بندی پیدا نشد.")

    output_path.write_text(
        json.dumps(chunks, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return output_path


if __name__ == "__main__":
    chunk_file(
        "synthetic_medibridge_healthcare_rag_dataset_clean.txt",
        max_words=250,
        overlap=40,
    )