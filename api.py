from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

import gradio as gr

import config
from app import build_demo
from ingest.embedder import EmbeddingError, get_model
from ingest.loaders import load_sources
from rag.chain import answer
from rag.llm import LLMError
from rag.logging_utils import get_logger
from rag.present import citation_payload, corpus_meta, index_ready, source_count, trace_payload
from rag.prompts import DISCLAIMER
from rag.retriever import IndexMissingError, RetrievalError
from rag.schemas import RefusalType

log = get_logger("api")

INDEX_UNAVAILABLE = (
    "search index unavailable; run `python -m ingest.build_index` and retry"
)
LLM_UNAVAILABLE = "generation unavailable; check LLM_API_KEY and LLM_MODEL"
ANSWER_FAILED = "failed to answer that question"
REDACTED_QUESTION = "[redacted: the question contained personal information]"


class AskRequest(BaseModel):
    question: str = Field(
        min_length=1,
        max_length=config.MAX_QUERY_CHARS,
        description="A factual question about one of the five HDFC Direct Growth schemes.",
        examples=["What is the expense ratio of HDFC Large Cap Fund - Direct Growth?"],
    )
    include_trace: bool = Field(
        default=True, description="Return the retrieval trace used to build the answer."
    )

    @field_validator("question")
    @classmethod
    def _reject_blank(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("question must not be blank")
        return cleaned


class CitationOut(BaseModel):
    source_url: str
    section: str
    scheme_name: str
    fetched_at: str


class AskResponse(BaseModel):
    question: str
    answer: str
    citations: list[CitationOut]
    last_updated: str
    refusal: bool
    refusal_type: str | None = None
    disclaimer: str = DISCLAIMER
    trace: dict[str, Any] | None = None


class HealthResponse(BaseModel):
    status: str
    index_ready: bool
    sources: int
    provider: str
    model: str
    collection: str
    disclaimer: str = DISCLAIMER
    corpus: dict[str, Any]


class SourceOut(BaseModel):
    scheme: str
    scheme_name: str
    category: str
    url: str


@asynccontextmanager
async def lifespan(_: FastAPI):
    if config.API_WARMUP:
        try:
            get_model()
            log.info("embedder_warm")
        except Exception as exc:
            log.warn("embedder_warm_failed", reason=str(exc))
    ready = index_ready()
    log.info(
        "startup",
        index_ready=ready,
        sources=source_count(),
        provider=config.LLM_PROVIDER,
        model=config.LLM_MODEL,
    )
    yield


app = FastAPI(
    title=config.API_TITLE,
    version=config.API_VERSION,
    summary="Grounded answers about five HDFC mutual fund Direct Growth schemes.",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(config.CORS_ORIGINS),
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse, tags=["ops"])
def health() -> HealthResponse:
    ready = index_ready()
    return HealthResponse(
        status="ok" if ready else "degraded",
        index_ready=ready,
        sources=source_count(),
        provider=config.LLM_PROVIDER,
        model=config.LLM_MODEL,
        collection=config.COLLECTION_NAME,
        corpus=corpus_meta(),
    )


@app.get("/sources", response_model=list[SourceOut], tags=["corpus"])
def sources() -> list[SourceOut]:
    return [
        SourceOut(
            scheme=row.scheme,
            scheme_name=row.scheme_name,
            category=row.category,
            url=row.url,
        )
        for row in load_sources()
    ]


@app.post("/ask", response_model=AskResponse, tags=["rag"])
def ask(request: AskRequest) -> AskResponse:
    question = request.question
    try:
        result = answer(question)
    except (IndexMissingError, RetrievalError) as exc:
        log.error("index_unavailable", reason=str(exc))
        raise HTTPException(status_code=503, detail=INDEX_UNAVAILABLE) from exc
    except LLMError as exc:
        log.error("generation_failed", reason=str(exc))
        raise HTTPException(status_code=503, detail=LLM_UNAVAILABLE) from exc
    except (EmbeddingError, OSError) as exc:
        log.error("embedding_failed", reason=f"{type(exc).__name__}: {exc}")
        raise HTTPException(status_code=503, detail=INDEX_UNAVAILABLE) from exc
    except Exception as exc:
        log.error("answer_failed", reason=f"{type(exc).__name__}: {exc}")
        raise HTTPException(status_code=500, detail=ANSWER_FAILED) from exc

    is_pii = result.refusal_type == RefusalType.PII.value
    return AskResponse(
        question=REDACTED_QUESTION if is_pii else question,
        answer=result.text,
        citations=[CitationOut(**citation) for citation in citation_payload(result)],
        last_updated=result.last_updated,
        refusal=result.refusal,
        refusal_type=result.refusal_type,
        trace=trace_payload(result) if request.include_trace else None,
    )


gr.mount_gradio_app(app, build_demo().queue(), path="/")


def main() -> None:
    import uvicorn

    uvicorn.run(
        "api:app",
        host=config.API_HOST,
        port=config.API_PORT,
        workers=config.API_WORKERS,
        log_level="info",
    )


if __name__ == "__main__":
    main()
