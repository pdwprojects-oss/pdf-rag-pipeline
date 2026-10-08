import re
from pathlib import Path


# فیلدهایی مثل FACILITY_ID، HEAD OFFICE و AFTER HOURS
FIELD_PATTERN = re.compile(
    r"(?<!\S)([A-Z][A-Z_]*(?:[ \t]+[A-Z][A-Z_]*)*)"
    r"[ \t]*[:=][ \t]*"
)

# عنوان بخش‌ها، مثل 01 COMPANY PROFILE
HEADING_PATTERN = re.compile(r"^\d{2}\s+[A-Z][A-Z0-9 /&_-]*$")

HEADER = "MEDIBRIDGE CARE NETWORK | SYNTHETIC RAG DATASET"
FOOTER = (
    "FICTIONAL DATA - FOR TEXT CLEANING / RAG PRACTICE ONLY"
    " - NOT A REAL HEALTHCARE DIRECTORY"
)

# شروع رکوردهای جدید
RECORD_STARTS = {
    "ENTITY_ID",
    "POLICY_ID",
    "FACILITY_ID",
    "MAPPING_ID",
    "DEPARTMENT",
    "FIELD",
    "QUERY_TOPIC",
}


def clean_text(text):
    lines = []

    for raw_line in text.splitlines():
        # حذف فاصله‌های اضافی، تب و فاصله‌های ابتدا و انتها
        line = re.sub(r"\s+", " ", raw_line).strip()

        if not line:
            continue

        if line in {HEADER, FOOTER}:
            continue

        if re.fullmatch(r"PAGE\s+\d+", line):
            continue

        if re.fullmatch(r"[=_-]{3,}", line):
            continue

        # حفظ عنوان بخش‌ها
        if HEADING_PATTERN.fullmatch(line):
            lines.append(("heading", line))
            continue

        matches = list(FIELD_PATTERN.finditer(line))

        if not matches:
            # اتصال ادامهٔ متن به فیلد قبلی، حتی در مرز صفحات
            if lines and lines[-1][0] == "field":
                kind, previous = lines[-1]
                lines[-1] = (kind, f"{previous} {line}")
            else:
                lines.append(("text", line))
            continue

        # متن قبل از اولین فیلد، اگر وجود داشته باشد
        prefix = line[:matches[0].start()].strip()

        if prefix:
            if lines and lines[-1][0] == "field":
                kind, previous = lines[-1]
                lines[-1] = (kind, f"{previous} {prefix}")
            else:
                lines.append(("text", prefix))

        # جداسازی فیلدهای هم‌خط و یکسان‌سازی : و =
        for index, match in enumerate(matches):
            key = match.group(1).strip()

            end = (
                matches[index + 1].start()
                if index + 1 < len(matches)
                else len(line)
            )

            value = line[match.end():end].strip()
            lines.append(("field", f"{key}: {value}"))

    # فاصله‌گذاری میان بخش‌ها و رکوردها
    output = []

    for kind, line in lines:
        key = line.split(":", 1)[0] if kind == "field" else ""

        starts_record = (
            key in RECORD_STARTS
            or key.startswith("NOISY_RECORD_")
        )

        if output and (kind == "heading" or starts_record):
            if output[-1] != "":
                output.append("")

        output.append(line)

        if kind == "heading":
            output.append("")

    return "\n".join(output).strip()


def clean_txt_file(filename):
    folder = Path(__file__).resolve().parent
    input_path = folder / filename

    if not input_path.is_file():
        raise FileNotFoundError(f"فایل پیدا نشد: {input_path}")

    # جلوگیری از نام‌هایی مثل clean_clean
    stem = input_path.stem.removesuffix("_clean")
    output_path = input_path.with_name(f"{stem}_clean.txt")

    if input_path.resolve() == output_path.resolve():
        raise ValueError(
            "فایل متنی اصلی Reader را وارد کن، نه فایل clean."
        )

    text = input_path.read_text(encoding="utf-8-sig")
    cleaned_text = clean_text(text)

    if not cleaned_text:
        raise ValueError("متن فایل پس از پاک‌سازی خالی است.")

    output_path.write_text(cleaned_text + "\n", encoding="utf-8")
    return output_path


if __name__ == "__main__":
    clean_txt_file("synthetic_medibridge_healthcare_rag_dataset.txt")