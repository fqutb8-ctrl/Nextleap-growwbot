from __future__ import annotations

import gradio as gr

from ingest.embedder import EmbeddingError, get_model
from rag.chain import answer
from rag.llm import LLMError
from rag.logging_utils import get_logger
from rag.present import BUILD_HINT, corpus_line, index_ready, trace_payload
from rag.prompts import DISCLAIMER
from rag.retriever import IndexMissingError, RetrievalError
from rag.schemas import RefusalType

log = get_logger("ui")

EXAMPLES = [
    "What is the expense ratio of the HDFC Large Cap Fund - Direct Growth?",
    "Is there a lock-in period in the HDFC ELSS Tax Saver Fund?",
    "What is the minimum SIP amount for HDFC Small Cap Fund - Direct Growth?",
]

WELCOME = (
    "Ask a factual question about the five HDFC Mutual Fund Direct Growth schemes "
    "below. Answers come only from the scheme pages in the corpus."
)

PII_PLACEHOLDER = "_[withheld: that question contained personal information]_"
LLM_HINT = (
    "Generation is unavailable. Check `LLM_API_KEY` and `LLM_MODEL` in your `.env` "
    "against the README, then restart the app."
)
ERROR_TEXT = "Something went wrong answering that question. Please try again."


def citation_html(result) -> str:
    if not result.citations:
        return ""
    lines = []
    for citation in result.citations:
        label = citation.section or citation.scheme_name or "source"
        lines.append(f"[{label}]({citation.source_url})")
    return "  \n".join(lines)


def footer(result) -> str:
    if not result.last_updated:
        return ""
    return f"Last updated from sources: {result.last_updated}"


def respond(question: str, history: list | None):
    history = list(history or [])
    question = (question or "").strip()
    if not question:
        return history, "", "", {}

    try:
        result = answer(question)
    except (IndexMissingError, RetrievalError) as exc:
        log.error("index_unavailable", reason=str(exc))
        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": BUILD_HINT})
        return history, "", "", {}
    except (EmbeddingError, OSError) as exc:
        log.error("embedding_failed", reason=str(exc))
        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": ERROR_TEXT})
        return history, "", "", {}
    except LLMError as exc:
        log.error("generation_failed", reason=str(exc))
        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": LLM_HINT})
        return history, "", "", {}
    except Exception as exc:
        log.error("answer_failed", reason=f"{type(exc).__name__}: {exc}")
        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": ERROR_TEXT})
        return history, "", "", {}

    if result.trace.guards.get("refusal_type") == RefusalType.PII.value:
        history.append({"role": "user", "content": PII_PLACEHOLDER})
    else:
        history.append({"role": "user", "content": question})
    history.append({"role": "assistant", "content": result.text})
    return history, citation_html(result), footer(result), trace_payload(result)


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="HDFC Mutual Fund FAQ") as demo:
        gr.Markdown(f"# HDFC Mutual Fund FAQ\n\n{WELCOME}\n\n**{DISCLAIMER}**")
        gr.Markdown(corpus_line())

        chatbot = gr.Chatbot(
            type="messages", label="Conversation", height=420, allow_tags=True
        )
        with gr.Row():
            question_box = gr.Textbox(
                label="Ask a question",
                placeholder="What is the expense ratio of HDFC Large Cap Fund?",
                scale=4,
            )
            send = gr.Button("Send", scale=1, variant="primary")

        gr.Examples(
            EXAMPLES,
            inputs=[question_box],
            label="Example questions",
        )

        citation = gr.Markdown(label="Source")
        updated = gr.Markdown()

        with gr.Accordion("Show retrieval trace", open=False):
            trace = gr.JSON(label="Trace")

        question_box.submit(
            respond,
            inputs=[question_box, chatbot],
            outputs=[chatbot, citation, updated, trace],
        ).then(lambda: "", None, question_box)
        send.click(
            respond,
            inputs=[question_box, chatbot],
            outputs=[chatbot, citation, updated, trace],
        ).then(lambda: "", None, question_box)

    return demo


def main() -> None:
    try:
        get_model()
        log.info("embedder_warm")
    except Exception as exc:
        log.warn("embedder_warm_failed", reason=str(exc))
    demo = build_demo()
    demo.queue().launch(server_name="127.0.0.1", server_port=7860, show_error=True)


if __name__ == "__main__":
    main()
