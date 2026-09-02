"""FastAPI server exposing the agent over SSE.

Deliberately a *separate* app from ``backend/main.py``. The source project's
FastAPI service is left byte-for-byte untouched so the before/after comparison
in the README stays honest — this one runs on a different port and shares no
code with it.

    python -m agent.server                    # then open http://127.0.0.1:8100

What the browser gets that the CLI also gets: token streaming (FR17), a status
indicator for every tool the model decides to call (FR18), and full conversation
memory (FR14). What it gets in addition is a trace panel, because in a demo the
interesting claim is not "the answer is right" but "the model chose these tools,
in this order, on its own".

Sessions are server-side. The conversation holds ``tool_use`` and ``tool_result``
blocks that cannot round-trip through a browser as JSON, and FR14 requires the
full history reach every model call — so the browser holds only a session id.
"""

from __future__ import annotations

import base64
import io
import json
import queue
import threading
import uuid
from pathlib import Path
from typing import Any

import anthropic
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel, Field

from agent import config
from agent.context import build_provider
from agent.loop import AgentLoop
from agent.memory import Conversation
from agent.sse import SSERenderer
from agent.tools.registry import build_dispatcher
from agent.trace import Tracer

app = FastAPI(
    title="小電 — Agentic Customer Service",
    description=(
        "An LLM-orchestrated customer-service agent. Claude decides at each step "
        "which tool to call — or whether to answer directly. Retrieval is one "
        "tool among five, not a mandatory preprocessing step."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------------
# Session store
# --------------------------------------------------------------------------


class Session:
    """One conversation, with its own memory, trace, and retrieval mode."""

    def __init__(self, session_id: str, mode: config.Mode, show_thinking: bool) -> None:
        self.id = session_id
        self.mode = mode
        self.show_thinking = show_thinking
        self.conversation = Conversation()
        self.tracer = Tracer(session_id, config.LOG_DIR, enabled=True)
        self.provider = build_provider(mode)
        self.dispatch, self.kb_size = build_dispatcher(
            mode, self.tracer, image_resolver=lambda ref: self.images.get(ref)
        )
        self.client = anthropic.Anthropic()
        # Uploaded images, keyed img_1, img_2, ... The agent never receives
        # binary; it gets an image_ref in the user turn and decides whether to
        # call the search tool with it.
        self.images: dict[str, Image.Image] = {}

        # One turn at a time per session — the conversation is mutable state.
        self.lock = threading.Lock()

    def add_image(self, data: bytes) -> str:
        ref = f"img_{len(self.images) + 1}"
        self.images[ref] = Image.open(io.BytesIO(data)).convert("RGB")
        return ref


SESSIONS: dict[str, Session] = {}


# --------------------------------------------------------------------------
# API models
# --------------------------------------------------------------------------


class NewSessionRequest(BaseModel):
    mode: str = Field(default=config.Mode.RAG.value)
    show_thinking: bool = False


class NewSessionResponse(BaseModel):
    session_id: str
    mode: str
    model: str
    kb_chunks: int | None


# --------------------------------------------------------------------------
# Endpoints
# --------------------------------------------------------------------------


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "model": config.MODEL,
        "modes": [m.value for m in config.Mode],
        "sessions": len(SESSIONS),
    }


@app.post("/api/session", response_model=NewSessionResponse)
def new_session(req: NewSessionRequest) -> NewSessionResponse:
    try:
        mode = config.Mode(req.mode)
    except ValueError:
        raise HTTPException(400, f"unknown mode: {req.mode}")

    session_id = f"web-{mode.value}-{uuid.uuid4().hex[:8]}"
    try:
        session = Session(session_id, mode, req.show_thinking)
    except FileNotFoundError as exc:
        # Mode A needs the KB index built; say so rather than 500ing.
        raise HTTPException(503, str(exc))

    SESSIONS[session_id] = session
    return NewSessionResponse(
        session_id=session_id,
        mode=mode.value,
        model=config.MODEL,
        kb_chunks=session.kb_size,
    )


@app.post("/api/chat")
async def chat(
    session_id: str = Form(...),
    message: str = Form(...),
    image: UploadFile | None = File(None),
) -> StreamingResponse:
    session = SESSIONS.get(session_id)
    if session is None:
        raise HTTPException(404, "工作階段不存在或已過期，請重新開始對話。")

    # An image with no text is a valid turn — the photo *is* the question.
    if image is not None:
        try:
            ref = session.add_image(await image.read())
        except Exception:
            raise HTTPException(400, "無法讀取上傳的圖片。")
        # Announce the attachment in the user turn. The model decides whether to
        # search with it, ask a clarifying question, or ignore it — the decision
        # stays with the agent rather than being forced by an if-statement.
        note = f"（顧客附上了一張商品照片，image_ref={ref}）"
        message = f"{message.strip()}\n{note}" if message.strip() else note
    elif not message.strip():
        raise HTTPException(400, "訊息不可為空。")

    sink: queue.Queue[dict[str, Any] | None] = queue.Queue()
    renderer = SSERenderer(sink, show_thinking=session.show_thinking)
    user_message = message

    loop = AgentLoop(
        client=session.client,
        provider=session.provider,
        dispatch=session.dispatch,
        renderer=renderer,
        tracer=session.tracer,
        conversation=session.conversation,
    )

    def run() -> None:
        """Drive the synchronous loop on a worker thread."""
        try:
            with session.lock:
                loop.run_turn(user_message)
            sink.put({"type": "done"})
        except anthropic.RateLimitError:
            sink.put({"type": "error", "text": "請求過於頻繁，請稍候再試。"})
        except anthropic.APIStatusError as exc:
            sink.put({"type": "error", "text": f"API 錯誤（{exc.status_code}）"})
        except anthropic.APIConnectionError:
            sink.put({"type": "error", "text": "連線失敗，請檢查網路。"})
        except Exception as exc:  # noqa: BLE001
            sink.put({"type": "error", "text": f"發生未預期的錯誤：{exc}"})
        finally:
            sink.put(None)  # sentinel: closes the stream

    threading.Thread(target=run, daemon=True).start()

    def events():
        while True:
            event = sink.get()
            if event is None:
                return
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/trace/{session_id}")
def trace(session_id: str) -> dict[str, Any]:
    """Structured trace for the demo's inspector panel (NFR3).

    This is the same JSONL the eval harness scores trajectories from — the point
    being that what the UI shows and what the evals assert are the same record.
    """
    session = SESSIONS.get(session_id)
    if session is None:
        raise HTTPException(404, "工作階段不存在。")
    path: Path = session.tracer.path
    if not path.exists():
        return {"records": []}

    keep = {"user_turn", "tool_call", "tool_result", "retrieval", "guard_tripped",
            "model_response", "image_search"}
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record["event"] in keep:
            records.append(record)
    return {"records": records}


@app.delete("/api/session/{session_id}")
def end_session(session_id: str) -> dict[str, str]:
    SESSIONS.pop(session_id, None)
    return {"status": "ended"}


# Mounted last so the API routes above take precedence.
_UI_DIR = config.BASE_DIR / "frontend_agent"
if _UI_DIR.exists():
    app.mount("/", StaticFiles(directory=str(_UI_DIR), html=True), name="ui")


def main() -> None:
    uvicorn.run(app, host="127.0.0.1", port=8100)


if __name__ == "__main__":
    main()
