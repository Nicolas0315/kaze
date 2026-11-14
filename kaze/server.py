"""OpenAI-compatible HTTP server (chat + completions, streaming)."""
from __future__ import annotations

import asyncio
import json
import time
import uuid
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from kaze.llm import LLM
from kaze.sampler import SamplingParams


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str = "kaze"
    messages: list[ChatMessage]
    max_tokens: int | None = None
    max_completion_tokens: int | None = None
    temperature: float = 0.7
    top_p: float = 1.0
    top_k: int = 0
    repetition_penalty: float = 1.0
    stop: list[str] | str | None = None
    stream: bool = False
    ignore_eos: bool = False
    seed: int | None = None


class CompletionRequest(BaseModel):
    model: str = "kaze"
    prompt: str | list[str]
    max_tokens: int = 128
    temperature: float = 0.7
    top_p: float = 1.0
    stop: list[str] | str | None = None
    stream: bool = False
    ignore_eos: bool = False


def _params(max_tokens: int, temperature: float, top_p: float, top_k: int,
            repetition_penalty: float, stop, ignore_eos: bool, seed) -> SamplingParams:
    stops = stop if isinstance(stop, list) else ([stop] if stop else [])
    return SamplingParams(max_tokens=max_tokens, temperature=temperature, top_p=top_p,
                          top_k=top_k or 0, repetition_penalty=repetition_penalty,
                          stop=[s for s in stops if s], ignore_eos=ignore_eos, seed=seed)


def create_app(llm: LLM, model_name: str = "kaze") -> FastAPI:
    app = FastAPI(title="Kaze", version="0.1.0")
    t_start = time.time()

    @app.get("/health")
    async def health() -> dict:
        return {"status": "ok", "uptime_s": round(time.time() - t_start, 1)}

    @app.get("/metrics")
    async def metrics() -> dict:
        return llm.metrics()

    @app.get("/v1/models")
    async def models() -> dict:
        return {"object": "list", "data": [
            {"id": model_name, "object": "model", "created": int(t_start), "owned_by": "kaze"}]}

    @app.post("/v1/chat/completions")
    async def chat(req: ChatRequest) -> Any:
        msgs = [{"role": m.role, "content": m.content} for m in req.messages]
        params = _params(req.max_tokens or req.max_completion_tokens or 256, req.temperature,
                         req.top_p, req.top_k, req.repetition_penalty, req.stop,
                         req.ignore_eos, req.seed)
        ids = llm._encode(msgs, chat=True)
        return await _serve(llm, ids, params, req.stream, model_name, chat=True)

    @app.post("/v1/completions")
    async def completions(req: CompletionRequest) -> Any:
        params = _params(req.max_tokens, req.temperature, req.top_p, 0, 1.0, req.stop,
                         req.ignore_eos, None)
        prompt = req.prompt[0] if isinstance(req.prompt, list) else req.prompt
        ids = llm._encode(prompt, chat=False)
        return await _serve(llm, ids, params, req.stream, model_name, chat=False)

    return app


async def _serve(llm: LLM, ids: list[int], params: SamplingParams, stream: bool,
                 model_name: str, chat: bool) -> Any:
    loop = asyncio.get_running_loop()
    q = llm.stream(ids, params)
    cid = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())
    if not stream:
        # the queue path beats a completion event at high concurrency (measured
        # 6,789 vs 5,638 tok/s at 32 streams): waking 32 threads at once makes them
        # fight over the GIL for tokenizer.decode + JSON.  The library API keeps the
        # event path (LLM.generate) because it wins at low concurrency.
        tokens: list[int] = []
        while True:
            out = await loop.run_in_executor(None, q.get)
            if out.tokens:
                tokens.extend(out.tokens)
            if out.finished:
                break
        text = llm.tokenizer.decode(tokens, skip_special_tokens=True)
        reason = out.finish_reason or "stop"
        if chat:
            return {"id": cid, "object": "chat.completion", "created": created,
                    "model": model_name,
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                                 "finish_reason": reason}],
                    "usage": {"prompt_tokens": len(ids), "completion_tokens": len(tokens),
                              "total_tokens": len(ids) + len(tokens)}}
        return {"id": cid, "object": "text_completion", "created": created, "model": model_name,
                "choices": [{"index": 0, "text": text, "finish_reason": reason}],
                "usage": {"prompt_tokens": len(ids), "completion_tokens": len(tokens),
                          "total_tokens": len(ids) + len(tokens)}}

    async def gen():
        tokens: list[int] = []
        sent = 0
        while True:
            out = await loop.run_in_executor(None, q.get)
            if out.tokens:
                tokens.extend(out.tokens)
                text = llm.tokenizer.decode(tokens, skip_special_tokens=True)
                delta = text[sent:]
                sent = len(text)
                if delta:
                    if chat:
                        payload = {"id": cid, "object": "chat.completion.chunk",
                                   "created": created, "model": model_name,
                                   "choices": [{"index": 0, "delta": {"content": delta},
                                                "finish_reason": None}]}
                    else:
                        payload = {"id": cid, "object": "text_completion", "created": created,
                                   "model": model_name,
                                   "choices": [{"index": 0, "text": delta, "finish_reason": None}]}
                    yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
            if out.finished:
                if chat:
                    payload = {"id": cid, "object": "chat.completion.chunk", "created": created,
                               "model": model_name,
                               "choices": [{"index": 0, "delta": {},
                                            "finish_reason": out.finish_reason or "stop"}]}
                else:
                    payload = {"id": cid, "object": "text_completion", "created": created,
                               "model": model_name,
                               "choices": [{"index": 0, "text": "",
                                            "finish_reason": out.finish_reason or "stop"}]}
                yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
                yield "data: [DONE]\n\n"
                break

    return StreamingResponse(gen(), media_type="text/event-stream")
