from pathlib import Path
from pypdf import PdfReader


def read_pdf_to_txt(filename):
    folder = Path(__file__).resolve().parent
    pdf_path = folder / filename
    txt_path = pdf_path.with_suffix(".txt")

    if not pdf_path.is_file():
        raise FileNotFoundError(f"فایل پیدا نشد: {pdf_path}")

    pages_text = []

    with pdf_path.open("rb") as file:
        reader = PdfReader(file)

        if reader.is_encrypted and not reader.decrypt(""):
            raise ValueError("فایل PDF رمز دارد.")

        for page in reader.pages:
            text = (page.extract_text() or "").strip()

            if text:
                pages_text.append(text)

    if not pages_text:
        raise ValueError("متنی پیدا نشد؛ شاید فایل به OCR نیاز دارد.")

    full_text = "\n\n".join(pages_text)
    txt_path.write_text(full_text, encoding="utf-8")

    return txt_path


if __name__ == "__main__":
    read_pdf_to_txt("synthetic_medibridge_healthcare_rag_dataset.pdf")