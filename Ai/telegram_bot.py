import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from .rag import answer_question
from .simplify import simplify_answer


ROOT = Path(__file__).resolve().parent.parent

# جلوگیری از ثبت URL درخواست‌ها که ممکن است توکن بات داشته باشند
logging.basicConfig(
    level=logging.WARNING,
    format="%(levelname)s: %(message)s",
)
logging.getLogger("httpx").setLevel(logging.CRITICAL)
logging.getLogger("httpcore").setLevel(logging.CRITICAL)

logger = logging.getLogger(__name__)


def is_private(update):
    return (
        update.effective_chat is not None
        and update.effective_chat.type == "private"
    )


async def check_access(update, context):
    if not is_private(update) or not update.effective_user:
        return False

    allowed_ids = context.application.bot_data["allowed_ids"]

    if update.effective_user.id not in allowed_ids:
        await update.effective_message.reply_text(
            "دسترسی فعال نیست. با /id شناسه‌ات را بگیر "
            "و آن را در تنظیمات برنامه قرار بده."
        )
        return False

    return True


async def show_id(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if is_private(update) and update.effective_user:
        await update.effective_message.reply_text(
            f"شناسهٔ تلگرام شما:\n{update.effective_user.id}"
        )


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await check_access(update, context):
        return

    await update.effective_message.reply_text(
        "سلام! دربارهٔ دیتاست MediBridge سؤال بپرس.\n"
        "اطلاعات این دیتاست ساختگی است.\n\n"
        "/reset — پاک‌کردن حافظه\n"
        "/simplify on — فعال‌کردن ساده‌سازی\n"
        "/simplify off — غیرفعال‌کردن ساده‌سازی\n"
        "/id — نمایش شناسه\n\n"
        "ساده‌سازی یک درخواست LLM اضافه مصرف می‌کند."
    )


async def reset(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await check_access(update, context):
        return

    context.chat_data.pop("history", None)

    await update.effective_message.reply_text(
        "حافظهٔ این چت پاک شد."
    )


async def toggle_simplify(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):
    if not await check_access(update, context):
        return

    option = context.args[0].lower() if context.args else ""

    if option not in {"on", "off"}:
        await update.effective_message.reply_text(
            "یکی از این دستورها را بفرست:\n"
            "/simplify on\n"
            "/simplify off"
        )
        return

    context.chat_data["simplify"] = option == "on"

    await update.effective_message.reply_text(
        "ساده‌سازی روشن شد."
        if option == "on"
        else "ساده‌سازی خاموش شد."
    )


async def send_answer(message, text):
    # تقسیم پاسخ‌های بلند؛ ارسال به‌صورت متن ساده
    for start_index in range(0, len(text), 1800):
        await message.reply_text(
            text[start_index:start_index + 1800],
            parse_mode=None,
        )


async def handle_message(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):
    if not await check_access(update, context):
        return

    message = update.effective_message
    question = (message.text or "").strip()

    if not question:
        return

    if len(question) > 2000:
        await message.reply_text(
            "لطفاً سؤال را کوتاه‌تر از ۲۰۰۰ کاراکتر بفرست."
        )
        return

    await message.reply_text("در حال بررسی سؤال…")

    try:
        history = context.chat_data.get("history", [])

        # این مسیر فایل last_answer.json را نمی‌نویسد
        result = await answer_question(
            question,
            history=history,
        )

        answer = result.get("answer")

        if not answer:
            await message.reply_text(
                "فعلاً پاسخی از مدل دریافت نشد. "
                "ممکن است محدودیت سرویس باشد؛ کمی بعد دوباره امتحان کن."
            )
            return

        simplification_failed = False

        if context.chat_data.get("simplify", False):
            try:
                simplified = await simplify_answer(answer)

                if simplified["error"] is None:
                    answer = simplified["text"]
                else:
                    simplification_failed = True

            except Exception:
                simplification_failed = True

        await send_answer(message, answer)

        # فقط پاسخ‌های موفق وارد حافظه می‌شوند
        history = history + [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer[:2000]},
        ]
        context.chat_data["history"] = history[-6:]

        if simplification_failed:
            await message.reply_text(
                "ساده‌سازی انجام نشد؛ پاسخ اصلی ارسال شد."
            )

    except Exception as exc:
        # متن خام خطا و اطلاعات اتصال را ثبت نمی‌کنیم
        logger.warning(
            "Question processing failed: %s",
            type(exc).__name__,
        )

        await message.reply_text(
            "پردازش سؤال با مشکل مواجه شد. "
            "اتصال سرویس و تنظیمات پروژه را بررسی کن."
        )


async def handle_error(
    update: object,
    context: ContextTypes.DEFAULT_TYPE,
):
    logger.warning(
        "Telegram operation failed: %s",
        type(context.error).__name__,
    )


def main():
    load_dotenv(ROOT / "knowledge" / ".env")

    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()

    if not token:
        raise ValueError("TELEGRAM_BOT_TOKEN تنظیم نشده است.")

    raw_ids = os.getenv("TELEGRAM_ALLOWED_USER_IDS", "")

    allowed_ids = {
        int(value.strip())
        for value in raw_ids.split(",")
        if value.strip()
    }

    # پردازش ترتیبی برای حفظ ترتیب حافظه و کاهش درخواست هم‌زمان
    application = (
        Application.builder()
        .token(token)
        .concurrent_updates(False)
        .build()
    )

    application.bot_data["allowed_ids"] = allowed_ids

    application.add_handler(CommandHandler("id", show_id))
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("reset", reset))
    application.add_handler(
        CommandHandler("simplify", toggle_simplify)
    )

    application.add_handler(
        MessageHandler(
            filters.TEXT & ~filters.COMMAND,
            handle_message,
        )
    )

    application.add_error_handler(handle_error)

    application.run_polling(
        allowed_updates=["message"],
        drop_pending_updates=True,
    )


if __name__ == "__main__":
    main()