"""Qwen3.8 系（Qwen3_5 text）を OpenAI 互換の HTTP で出す小さなサーバ。

dense 用の `kaze.server` は KV キャッシュ前提のエンジンに合わせてあるので、
こちらは `generate` / `stream_tokens` を直接叩く薄い実装にしています。
"""
from __future__ import annotations

import json
import time
from typing import Any

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from kaze.qwen38 import stream_tokens


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str = "qwen3.8"
    messages: list[ChatMessage]
    max_tokens: int = Field(256, ge=1, le=8192)
    temperature: float = Field(0.0, ge=0.0, le=2.0)
    stream: bool = False


class CompletionRequest(BaseModel):
    model: str = "qwen3.8"
    prompt: str = ""
    max_tokens: int = Field(256, ge=1, le=8192)
    temperature: float = Field(0.0, ge=0.0, le=2.0)
    stream: bool = False


def _render(tok, messages: list[dict], thinking: bool = False) -> str:
    try:
        return tok.apply_chat_template(messages, tokenize=False,
                                       add_generation_prompt=True,
                                       enable_thinking=thinking)
    except Exception:
        return tok.apply_chat_template(messages, tokenize=False,
                                       add_generation_prompt=True)


def create_app(model, tok, model_name: str = "qwen3.8", graph: bool = False,
               max_len: int = 4096) -> FastAPI:
    import threading

    from kaze.qwen38 import GraphDecoder

    app = FastAPI(title=f"kaze ({model_name})")
    # グラフは起動時に 1 回だけキャプチャする（リクエスト中の別スレッドで捕まえると壊れる）
    decoder = GraphDecoder(model, max_len=max_len) if graph else None
    lock = threading.Lock()
    eos = (tok.eos_token_id,) if getattr(tok, "eos_token_id", None) is not None else ()

    @app.get("/health")
    async def health() -> dict:
        return {"ok": True, "model": model_name}

    @app.get("/v1/models")
    async def models() -> dict:
        return {"object": "list", "data": [{"id": model_name, "object": "model"}]}

    @app.post("/v1/chat/completions")
    async def chat(req: ChatRequest) -> Any:
        text = _render(tok, [m.model_dump() for m in req.messages])
        return _respond(req, text, tok, model, eos, graph, max_len, chat=True,
                        decoder=decoder, lock=lock)

    @app.post("/v1/completions")
    async def completions(req: CompletionRequest) -> Any:
        return _respond(req, req.prompt, tok, model, eos, graph, max_len, chat=False,
                        decoder=decoder, lock=lock)

    return app


def _respond(req, text: str, tok, model, eos, graph, max_len, chat: bool,
             decoder=None, lock=None) -> Any:
    ids = tok(text).input_ids
    created = int(time.time())
    name = getattr(req, "model", "qwen3.8")
    it = stream_tokens(model, ids, max_tokens=req.max_tokens,
                       temperature=req.temperature, eos_token_ids=eos,
                       graph=graph, max_len=max_len, decoder=decoder)
    if req.stream:
        def gen():
            for i, token in enumerate(it):
                piece = tok.decode([token], skip_special_tokens=True)
                payload = {"choices": [{"index": 0, "delta": {"content": piece}}]}
                yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(gen(), media_type="text/event-stream")

    if lock is not None:
        with lock:
            out = list(it)
    else:
        out = list(it)
    content = tok.decode(out, skip_special_tokens=True)
    if chat:
        payload = {
            "id": f"chatcmpl-{created}",
            "object": "chat.completion",
            "created": created,
            "model": name,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": content},
                        "finish_reason": "stop"}],
            "usage": {"completion_tokens": len(out)},
        }
    else:
        payload = {
            "id": f"cmpl-{created}",
            "object": "text_completion",
            "created": created,
            "model": name,
            "choices": [{"index": 0, "text": content, "finish_reason": "stop"}],
            "usage": {"completion_tokens": len(out)},
        }
    return payload
