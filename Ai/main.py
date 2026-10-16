import argparse
import asyncio
import json
from pathlib import Path

from dotenv import load_dotenv

from .fallback_manager import error_reason
from .rag import answer_question
from .simplify import simplify_answer


ROOT = Path(__file__).resolve().parent.parent
OUTPUT_PATH = ROOT / "knowledge" / "last_answer.json"


async def run(question, simplify=False):
    load_dotenv(ROOT / "knowledge" / ".env")

    try:
        result = await answer_question(question)

        if simplify and result["answer"]:
            simplified = await simplify_answer(result["answer"])

            result["original_answer"] = result["answer"]
            result["answer"] = simplified["text"]
            result["simplification"] = {
                "model": simplified["model"],
                "provider": simplified["provider"],
                "attempts": simplified["attempts"],
                "error": simplified["error"],
            }

    except Exception as exc:
        result = {
            "question": question,
            "answer": None,
            "error": error_reason(exc),
            "retrieved_chunks": [],
        }

    OUTPUT_PATH.write_text(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("question")
    parser.add_argument("--simplify", action="store_true")
    args = parser.parse_args()

    asyncio.run(run(args.question, args.simplify))