# PDF RAG Pipeline & Telegram Bot

A Python learning project for document-based question answering
using a synthetic MediBridge healthcare dataset.

## Features

- PDF text extraction and dataset-specific cleaning
- Chunking with source metadata
- Vector search using cosine similarity
- Answers with chunk citations
- Optional answer simplification
- Model fallback on errors, empty responses, or timeouts
- Telegram bot with short conversation memory
- Telegram user allowlist

## Requirements

- Python 3.10+
- A GapGPT API key and sufficient account credit
- A Telegram bot token for Telegram mode
- Internet access to the required services
- An optional Google Gemini API key

## Setup

Run commands from the repository root:

```bash
python -m pip install -r knowledge/requirements.txt
```

Copy `knowledge/.env.example` to `knowledge/.env`.

Set your GapGPT API key and chat models:

```env
GEMINI_API_KEY=
GAPGPT_API_KEY=YOUR_GAPGPT_API_KEY
GAPGPT_CHAT_MODEL=gpt-6-luna
GAPGPT_FALLBACK_MODEL=gpt-5.6-luna
TELEGRAM_BOT_TOKEN=YOUR_TELEGRAM_BOT_TOKEN
TELEGRAM_ALLOWED_USER_IDS=
```

Keep real credentials in `.env` only.

## Prepare the data

The dataset and generated files are not included.

Place this text-based PDF inside `knowledge/`:

`synthetic_medibridge_healthcare_rag_dataset.pdf`

Then run these commands in order. Stop if a command fails:

```bash
python knowledge/reader.py
python knowledge/cleaning.py
python knowledge/chunking.py
python knowledge/embedding.py
```

For a different dataset, update the filenames and cleaning rules.

Document and query embeddings both use
`text-embedding-3-small` through GapGPT.

If you change the embedding model, regenerate document embeddings.

Sections 13 and 14 of the MediBridge dataset are excluded
from retrieval because they contain sample answers and cleaning tests.

## Ask a question

```bash
python -m Ai.main "Which Lakeside pharmacy offers drive-up pickup?"
```

Enable optional simplification:

```bash
python -m Ai.main "Which Lakeside pharmacy offers drive-up pickup?" --simplify
```

CLI results are saved to `knowledge/last_answer.json`.

Simplification uses an additional LLM request.

## Telegram bot

Create a bot through the official Telegram BotFather and
set `TELEGRAM_BOT_TOKEN` in `.env`.

Start the bot:

```bash
python -m Ai.telegram_bot
```

Send `/id` to your bot in a private chat.

Stop the process and set the returned numeric ID:

```env
TELEGRAM_ALLOWED_USER_IDS=123456789
```

Restart the bot. Multiple allowed IDs can be separated by commas.

An empty allowlist does not allow LLM requests.

### Commands

- `/start`: show help
- `/id`: show your Telegram user ID
- `/reset`: clear conversation history
- `/simplify on`: enable simplification
- `/simplify off`: disable simplification

The bot keeps up to three previous question-answer pairs per chat.
Memory is stored in RAM and is lost on restart.

Messages are processed sequentially.
Telegram replies are sent directly, without writing the CLI output file.

## Model fallback

With `GEMINI_API_KEY` empty, the default order is:

`gpt-6-luna → gpt-5.6-luna`

If a Google key is configured, the Gemini models listed in
`Ai/fallback_manager.py` are attempted first.

Each answer-generation model has a 10-second timeout.
Query embedding has a separate 15-second timeout.

Fallback does not guarantee success during account-wide limits
or provider outages. Model availability and pricing depend on
the provider. Client cancellation may not stop server-side processing.

## Limitations and privacy

- This is a learning prototype, not a medical advice system.
- The MediBridge dataset is fictional.
- Scanned PDFs require OCR, which is not included.
- Conversation memory and retrieval can make mistakes.
- Retrieved text and questions are sent to the configured AI services.
- Telegram messages also pass through Telegram.
- Source documents, generated outputs, and credentials are excluded from Git.
- Publishing this repository does not keep the bot running.
- Continuous operation requires a running computer or a separately hosted service.

## Validation performed

- Document retrieval and cited answers
- Follow-up questions in Telegram
- Optional answer simplification
- Fallback after an intentionally invalid model returned an error

The 10-second timeout behavior has not been separately tested.