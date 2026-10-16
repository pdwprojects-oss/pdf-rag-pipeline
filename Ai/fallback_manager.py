import asyncio
import os

from google import genai
from google.genai import types
from openai import AsyncOpenAI


GAPGPT_URL = "https://api.gapgpt.app/v1"

GEMINI_MODELS = [
    ("gemini-3.8-flash", 10),
    ("gemini-3.6-flash", 10),
    ("gemini-3.5-flash-lite", 10),
]

GAPGPT_TIMEOUT = 10


def error_reason(exc):
    """ثبت نوع خطا بدون ذخیرهٔ کلیدها یا متن کامل درخواست."""
    if isinstance(exc, asyncio.TimeoutError):
        return "timeout"

    code = getattr(exc, "status_code", None)

    if code is None:
        code = getattr(exc, "code", None)

    if code == 429:
        body = getattr(exc, "body", None)
        message = ""

        if isinstance(body, dict):
            error = body.get("error", body)

            if isinstance(error, dict):
                message = str(error.get("message", ""))
            elif isinstance(error, str):
                message = error

        if not message:
            message = "توضیح بیشتری از سرویس دریافت نشد."

        # حذف کلیدها، اگر در توضیح خطا وجود داشته باشند
        for name in ("GAPGPT_API_KEY", "GEMINI_API_KEY"):
            key = os.getenv(name, "").strip()

            if key:
                message = message.replace(key, "[REDACTED]")

        response = getattr(exc, "response", None)
        headers = getattr(response, "headers", {})
        retry_after = headers.get("retry-after")

        return (
            f"api_error_429: {message[:800]}"
            f" | retry_after={retry_after}"
        )

    if code is not None:
        return f"api_error_{code}"

    return type(exc).__name__


def successful_result(text, model, provider, attempts):
    return {
        "text": text,
        "model": model,
        "provider": provider,
        "attempts": attempts,
        "error": None,
    }


async def generate_text(prompt, instruction):
    attempts = []

    # مرحلهٔ اول: مدل‌های مستقیم گوگل
    google_key = os.getenv("GEMINI_API_KEY", "").strip()

    if google_key:
        try:
            async with genai.Client(
                api_key=google_key,
                http_options=types.HttpOptions(
                    timeout=10000,
                    retry_options=types.HttpRetryOptions(
                        attempts=1,
                    ),
                ),
            ).aio as client:
                for model, seconds in GEMINI_MODELS:
                    try:
                        response = await asyncio.wait_for(
                            client.models.generate_content(
                                model=model,
                                contents=prompt,
                                config=types.GenerateContentConfig(
                                    system_instruction=instruction,
                                    max_output_tokens=1200,
                                ),
                            ),
                            timeout=seconds,
                        )

                        text = (response.text or "").strip()

                        if text:
                            return successful_result(
                                text=text,
                                model=model,
                                provider="google",
                                attempts=attempts,
                            )

                        reason = "empty_response"

                    except Exception as exc:
                        reason = error_reason(exc)

                    attempts.append({
                        "provider": "google",
                        "model": model,
                        "reason": reason,
                    })

        except Exception as exc:
            attempts.append({
                "provider": "google",
                "reason": error_reason(exc),
            })

    # مرحلهٔ دوم: مدل اصلی و جایگزین GapGPT
    gap_key = os.getenv("GAPGPT_API_KEY", "").strip()

    # حذف نام‌های خالی و تکراری، با حفظ ترتیب
    gap_models = list(dict.fromkeys(
        model.strip()
        for model in [
            os.getenv("GAPGPT_CHAT_MODEL", ""),
            os.getenv("GAPGPT_FALLBACK_MODEL", ""),
        ]
        if model.strip()
    ))

    if gap_key and gap_models:
        try:
            async with AsyncOpenAI(
                base_url=GAPGPT_URL,
                api_key=gap_key,
                timeout=GAPGPT_TIMEOUT,
                max_retries=0,
            ) as client:
                for model in gap_models:
                    try:
                        response = await asyncio.wait_for(
                            client.chat.completions.create(
                                model=model,
                                messages=[
                                    {
                                        "role": "system",
                                        "content": instruction,
                                    },
                                    {
                                        "role": "user",
                                        "content": prompt,
                                    },
                                ],
                                max_tokens=1200,
                            ),
                            timeout=GAPGPT_TIMEOUT,
                        )

                        text = (
                            response.choices[0].message.content or ""
                        ).strip() if response.choices else ""

                        if text:
                            return successful_result(
                                text=text,
                                model=model,
                                provider="gapgpt",
                                attempts=attempts,
                            )

                        reason = "empty_response"

                    except Exception as exc:
                        reason = error_reason(exc)

                    attempts.append({
                        "provider": "gapgpt",
                        "model": model,
                        "reason": reason,
                    })

        except Exception as exc:
            attempts.append({
                "provider": "gapgpt",
                "reason": error_reason(exc),
            })

    else:
        attempts.append({
            "provider": "gapgpt",
            "reason": "missing_api_key_or_model",
        })

    return {
        "text": None,
        "model": None,
        "provider": None,
        "attempts": attempts,
        "error": "هیچ‌کدام از مدل‌ها پاسخ قابل استفاده ندادند.",
    }