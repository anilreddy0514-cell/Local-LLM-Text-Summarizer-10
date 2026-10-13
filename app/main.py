import os
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, field_validator

BASE_DIR = Path(__file__).resolve().parent
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "tinyllama")
OLLAMA_TIMEOUT = float(os.getenv("OLLAMA_TIMEOUT", "120"))
MAX_TEXT_LENGTH = 20_000

app = FastAPI(title="Local Text Summarizer", version="1.0.0")


class SummarizeRequest(BaseModel):
    text: str = Field(min_length=1, max_length=MAX_TEXT_LENGTH)

    @field_validator("text")
    @classmethod
    def text_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Text must contain at least one non-whitespace character.")
        return value.strip()


class SummarizeResponse(BaseModel):
    summary: str


@app.get("/", include_in_schema=False)
async def index() -> FileResponse:
    return FileResponse(BASE_DIR / "static" / "index.html")


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "model": OLLAMA_MODEL}


@app.post("/api/summarize", response_model=SummarizeResponse)
async def summarize(request: SummarizeRequest) -> SummarizeResponse:
    prompt = (
        "Summarize the text below in 2 to 4 concise sentences. Preserve its key "
        "facts and meaning. Do not add information that is not in the text.\n\n"
        f"Text:\n{request.text}\n\nSummary:"
    )
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.3, "num_predict": 180},
    }

    try:
        async with httpx.AsyncClient(timeout=OLLAMA_TIMEOUT) as client:
            response = await client.post(f"{OLLAMA_URL}/api/generate", json=payload)
            response.raise_for_status()
    except httpx.TimeoutException as exc:
        raise HTTPException(
            status_code=504,
            detail="Ollama took too long to respond. Try a shorter text or try again.",
        ) from exc
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 404:
            detail = f"Ollama model '{OLLAMA_MODEL}' was not found. Pull it with `ollama pull {OLLAMA_MODEL}`."
        else:
            detail = "Ollama returned an error. Check that the local Ollama service is running."
        raise HTTPException(status_code=502, detail=detail) from exc
    except httpx.RequestError as exc:
        raise HTTPException(
            status_code=503,
            detail="Could not connect to Ollama. Start Ollama and try again.",
        ) from exc

    try:
        summary = response.json().get("response", "").strip()
    except (ValueError, AttributeError) as exc:
        raise HTTPException(
            status_code=502,
            detail="Ollama returned an invalid response. Please try again.",
        ) from exc

    if not summary:
        raise HTTPException(
            status_code=502,
            detail="Ollama returned an empty summary. Please try again.",
        )
    return SummarizeResponse(summary=summary)
