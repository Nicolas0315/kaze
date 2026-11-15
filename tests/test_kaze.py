"""Kaze test suite (no pytest required: python tests/test_kaze.py)."""
from __future__ import annotations

import os
import sys
import traceback

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kaze.cache import KVCache
from kaze.config import ModelConfig
from kaze.engine import Engine, SamplingParams
from kaze.kernels import attn as K_attn
from kaze.kernels import quant_linear as K_lin
from kaze.quant import QuantLinear, dequantize_int4, quantize_int4

DEV = "cuda"
MODELS = os.environ.get("KAZE_MODELS", os.path.expanduser("~/models"))
MODEL = os.path.join(MODELS, "Qwen2.5-0.5B-Instruct")
_results: list[tuple[str, bool, str]] = []


def check(name: str):
    def deco(fn):
        try:
            fn()
            _results.append((name, True, ""))
            print(f"PASS {name}")
        except Exception as exc:                    # noqa: BLE001
            _results.append((name, False, str(exc)))
            print(f"FAIL {name}: {exc}")
            traceback.print_exc(limit=3)
        return fn
    return deco


@check("quant.int4_roundtrip")
def _() -> None:
    torch.manual_seed(0)
    w = torch.randn(64, 512, device=DEV) * 0.02
    packed, scales = quantize_int4(w, 128)
    dq = dequantize_int4(packed, scales, 128)
    rel = (dq - w).norm() / w.norm()
    assert rel < 0.12, rel


@check("kernel.gemv_int8_matches_dequant")
def _() -> None:
    torch.manual_seed(0)
    n, k = 512, 896
    w = torch.randn(n, k, device=DEV) * 0.02
    lin = QuantLinear.from_float(w, torch.randn(n, device=DEV), kind="int8", group=128)
    x = torch.zeros(16, k, device=DEV, dtype=torch.float16)
    x[0] = torch.randn(k, device=DEV)
    got = K_lin.gemv(x, lin, 16)[0].float()
    ref = lin.dequantize().float() @ x[0].float() + lin.bias.float()
    assert torch.nn.functional.cosine_similarity(got, ref, dim=0) > 0.999


@check("kernel.gemm_int8_matches_dequant")
def _() -> None:
    torch.manual_seed(0)
    n, k, m = 512, 896, 37
    w = torch.randn(n, k, device=DEV) * 0.02
    lin = QuantLinear.from_float(w, None, kind="int8", group=128)
    a = torch.randn(m, k, device=DEV, dtype=torch.float16)
    got = K_lin.gemm(a, lin).float()
    ref = a.float() @ lin.dequantize().float().T
    rel = (got - ref).norm() / ref.norm()
    assert rel < 0.01, rel


@check("kernel.paged_attention_matches_reference")
def _() -> None:
    torch.manual_seed(1)
    b, hq, hkv, d, bs, nb = 2, 14, 2, 64, 16, 6
    gqa = hq // hkv
    kc = torch.randn(nb, bs, hkv, d, device=DEV, dtype=torch.float16) * 0.3
    vc = torch.randn(nb, bs, hkv, d, device=DEV, dtype=torch.float16) * 0.3
    q = torch.randn(b, hq, d, device=DEV, dtype=torch.float16) * 0.3
    lens = torch.tensor([35, 20], device=DEV, dtype=torch.int32)
    bt = torch.zeros(b, 8, dtype=torch.int32, device=DEV)
    bt[0, :3] = torch.tensor([5, 2, 4], device=DEV)
    bt[1, :2] = torch.tensor([1, 0], device=DEV)
    out = torch.zeros(b, hq, d, device=DEV, dtype=torch.float16)
    acc = torch.zeros(b, hkv, 8, 16, d, device=DEV, dtype=torch.float32)
    ml = torch.zeros(b, hkv, 8, 2, 16, device=DEV, dtype=torch.float32)
    K_attn.attn_decode(q, kc, vc, bt, lens, bs, gqa, d, out, acc, ml, scale=d ** -0.5)
    kf = kc.view(-1, hkv, d)
    vf = vc.view(-1, hkv, d)
    for i in range(b):
        ln = int(lens[i])
        slots = torch.tensor([bt[i, t // bs].item() * bs + t % bs for t in range(ln)], device=DEV)
        K = kf.index_select(0, slots)
        V = vf.index_select(0, slots)
        for h in range(hq):
            kvh = h // gqa
            s = (q[i, h].float() @ K[:, kvh].float().T) * d ** -0.5
            ref = torch.softmax(s, -1) @ V[:, kvh].float()
            cos = torch.nn.functional.cosine_similarity(ref, out[i, h].float(), dim=0).item()
            assert cos > 0.999, (i, h, cos)


def _load_model(kind: str = "int8"):
    from kaze.loader import load_model
    return load_model(MODEL, kind=kind, group=128, lm_head_kind="int8", verbose=False)


@check("model.decode_equals_prefill")
def _() -> None:
    model = _load_model("fp16")
    dev = model.device
    cache = KVCache(model.cfg, 64, 16, dev)
    ids = torch.randint(0, 30000, (23,), device=dev)
    blocks = [cache.allocator.allocate() for _ in range(2)]
    lg = model.prefill_logits(ids, torch.arange(23, device=dev), cache,
                              cache.block_table_tensor(blocks, 8), 0)
    tok = int(lg[0].argmax())
    sl = cache.slots(blocks, 23, 24)
    bt = cache.block_table_tensor(blocks, 8)
    dec = model.decode_logits(torch.tensor([tok], device=dev), torch.tensor([23], device=dev),
                              sl, bt, torch.tensor([24], device=dev, dtype=torch.int32),
                              cache, tdim=16)
    ids2 = torch.cat([ids, torch.tensor([tok], device=dev)])
    blocks2 = [cache.allocator.allocate() for _ in range(2)]
    ref = model.prefill_logits(ids2, torch.arange(24, device=dev), cache,
                               cache.block_table_tensor(blocks2, 8), 0)
    cos = torch.nn.functional.cosine_similarity(dec[0].float(), ref[0].float(), dim=0).item()
    assert cos > 0.9999, cos


@check("engine.chunked_prefill_and_prefix_cache")
def _() -> None:
    model = _load_model("int8")
    prompt = list(torch.randint(0, 30000, (61,)).tolist())
    prompts = [prompt, prompt[:32] + list(torch.randint(0, 30000, (29,)).tolist())]

    def run(prefix_cache: bool, max_tokens: int = 8) -> list[list[int]]:
        eng = Engine(model, max_batch=8, max_batched_tokens=16, block_size=16,
                     use_cuda_graph=False, enable_prefix_cache=prefix_cache, verbose=False,
                     num_blocks=256)
        outs = []
        for p in prompts:
            r = eng.generate(p, SamplingParams(max_tokens=max_tokens, temperature=0.0))
            outs.append(r.tokens)
        return outs

    a = run(False)
    b = run(True)
    assert a == b, (a, b)


@check("engine.concurrent_batching")
def _() -> None:
    model = _load_model("int8")
    eng = Engine(model, max_batch=8, block_size=16, use_cuda_graph=False, verbose=False,
                 num_blocks=256)
    eng.capture_graphs()
    params = [SamplingParams(max_tokens=8, temperature=0.0) for _ in range(4)]
    seqs = [eng.add_request(list(torch.randint(0, 30000, (20,)).tolist()), p) for p in params]
    eng.run_until_idle()
    assert all(s.state == 3 for s in seqs)
    assert all(len(s.output) == 8 for s in seqs)


@check("engine.stop_and_length")
def _() -> None:
    model = _load_model("int8")
    eng = Engine(model, max_batch=4, block_size=16, use_cuda_graph=False, verbose=False,
                 num_blocks=128)
    r = eng.generate(list(torch.randint(0, 30000, (12,)).tolist()),
                     SamplingParams(max_tokens=5, temperature=0.0))
    assert len(r.tokens) == 5 and r.finish_reason == "length"


@check("sampler.greedy_deterministic")
def _() -> None:
    from kaze.sampler import sample_rows
    logits = torch.randn(4, 100, device=DEV)
    p = [SamplingParams(temperature=0.0) for _ in range(4)]
    a = sample_rows(logits, p)
    b = sample_rows(logits.clone(), p)
    assert torch.equal(a, b)


@check("sampler.top_k_restricts")
def _() -> None:
    from kaze.sampler import sample_rows
    logits = torch.zeros(1, 1000, device=DEV)
    logits[0, 5] = 10.0
    p = [SamplingParams(temperature=1.0, top_k=1)]
    for _ in range(8):
        assert int(sample_rows(logits, p)[0]) == 5


def main() -> int:
    bad = [r for r in _results if not r[1]]
    print(f"\n{len(_results) - len(bad)}/{len(_results)} passed")
    return 1 if bad else 0


# ---------------------------------------------------------------- MoE
def _moe_model():
    import os
    p = os.path.join(MODELS, "tiny-qwen3moe")
    if not os.path.isdir(p):
        return None
    from kaze.loader import load_model
    return load_model(p, kind="fp16", verbose=False)


@check("moe.block_matches_manual")
def test_moe_block_matches_manual() -> None:
    """Router + top-k + expert GEMV must match a plain torch computation."""
    from kaze.kernels import quant_linear as K_lin
    model = _moe_model()
    if model is None:
        return
    layer = model.layers[0]
    torch.manual_seed(0)
    x = torch.randn(5, model.cfg.hidden_size, device=model.device, dtype=torch.float16) * 0.5
    i, h, e_n = layer.moe_i, model.cfg.hidden_size, model.cfg.num_experts
    k = model.cfg.num_experts_per_tok
    logits = torch.nn.functional.linear(x.float(), layer.router.float())
    w, idx = torch.topk(logits, k, dim=-1)
    if model.cfg.norm_topk_prob:
        w = w / w.sum(-1, keepdim=True)
    ref = torch.zeros(5, h, dtype=torch.float16, device=x.device)
    gu_all, dn_all = layer.w_gu_e.packed, layer.w_down_e.packed
    for t in range(5):
        acc = torch.zeros(h, dtype=torch.float32, device=x.device)
        for kk in range(k):
            e = int(idx[t, kk])
            gu = gu_all[e * 2 * i:(e + 1) * 2 * i] @ x[t]
            g, u = gu[:i], gu[i:]
            act = g.float() * torch.sigmoid(g.float()) * u.float()
            acc += (dn_all[e * h:(e + 1) * h] @ act.half()).float() * w[t, kk]
        ref[t] = acc.half()
    mine = torch.zeros(5, h, dtype=torch.float16, device=x.device)
    for e in range(e_n):
        sel = idx == e
        rows = sel.any(-1).nonzero(as_tuple=True)[0]
        if rows.numel() == 0:
            continue
        xe = x.index_select(0, rows)
        gu = K_lin.gemm(xe, layer.w_gu_e, row_off=e * 2 * i, n_override=2 * i)
        g, u = gu[:, :i], gu[:, i:]
        dn = K_lin.gemm(g * torch.sigmoid(g) * u, layer.w_down_e, row_off=e * h, n_override=h)
        wsel = (sel[rows].to(torch.float32) * w[rows]).sum(-1, keepdim=True)
        mine.index_add_(0, rows, dn * wsel.to(dn.dtype))
    assert torch.allclose(ref, mine, atol=3e-2), (ref - mine).abs().max().item()


@check("moe.decode_equals_prefill")
def test_moe_decode_equals_prefill() -> None:
    """The decode path (moe_gemv) must agree with the prefill path."""
    from kaze.cache import KVCache
    model = _moe_model()
    if model is None:
        return
    cfg = model.cfg
    ids = torch.tensor([1, 5, 9, 20, 300, 42], device=model.device)
    pos = torch.arange(ids.numel(), device=model.device)
    cache = KVCache(cfg, 64, 8, model.device)
    bt = cache.block_table_tensor([cache.allocator.allocate() for _ in range(4)], 8)
    full = model.prefill_logits(ids, pos, cache, bt, 0, last_only=False)
    cache2 = KVCache(cfg, 64, 8, model.device)
    bt2 = cache2.block_table_tensor([cache2.allocator.allocate() for _ in range(4)], 8)
    rows = [model.prefill_logits(ids[i:i + 1], pos[i:i + 1], cache2, bt2, i)[0]
            for i in range(ids.numel())]
    dec = torch.stack(rows)
    cos = torch.nn.functional.cosine_similarity(full.float(), dec.float(), dim=-1).min().item()
    assert cos > 0.999, cos
    assert int((full.argmax(-1) == dec.argmax(-1)).sum()) == ids.numel()


@check("moe.engine_cuda_graph")
def test_moe_engine_graph() -> None:
    """A MoE model must run through the CUDA-graph decode path and match eager."""
    from kaze.engine import Engine, SamplingParams
    model = _moe_model()
    if model is None:
        return
    ids = list(torch.randint(0, 30000, (12,)).tolist())
    outs = []
    for graph in (False, True):
        eng = Engine(model, max_batch=2, use_cuda_graph=graph, speculative=False,
                     verbose=False, num_blocks=64)
        if graph:
            eng.capture_graphs()
        r = eng.generate(ids, SamplingParams(max_tokens=6, temperature=0.0))
        outs.append(r.tokens)
    assert outs[0] == outs[1], outs


@check("engine.speculative_multi")
def test_speculative_multi() -> None:
    """Speculation on a batch must not change the greedy output."""
    from kaze.engine import Engine, SamplingParams
    model = _load_model("int8")
    # repeating cycles so the n-gram drafter always has something to propose
    cycle = [11, 22, 33, 44, 55, 66]
    # each prompt ends on a 3-gram that already occurred, so the drafter has a match
    prompts = [list(cycle) * 3 + [11, 22, 33, 200 + i] for i in range(4)]
    out = []
    spec_steps = []
    for spec in (False, True):
        eng = Engine(model, max_batch=8, use_cuda_graph=False, speculative=spec,
                     spec_k=5, verbose=False, num_blocks=256, spec_accept_ema=4.0)
        # this test exercises the multi-sequence verification path on purpose
        eng.spec_max_batch = 8
        toks = []
        seqs = [eng.add_request(p, SamplingParams(max_tokens=12, temperature=0.0))
                for p in prompts]
        while any(s.state != 3 for s in seqs):
            eng.step()
        spec_steps.append(eng.stats["spec_steps"])
        for s in seqs:
            toks.append(list(s.output))
        out.append(toks)
    assert out[0] == out[1], out
    assert spec_steps[1] > 0, "multi-sequence speculation never ran"


@check("spec.verify_matches_decode")
def test_verify_matches_decode() -> None:
    """The verification forward must agree with stepping one token at a time."""
    from kaze.cache import KVCache
    model = _load_model("int8")
    cfg = model.cfg
    torch.manual_seed(0)
    prompt = torch.randint(0, 3000, (18,), device=model.device)
    draft = torch.randint(0, 3000, (4,), device=model.device)
    K = 5
    ref = []
    cache = KVCache(cfg, 64, 16, model.device)
    bt = cache.block_table_tensor([cache.allocator.allocate() for _ in range(4)], 4)
    model.prefill_logits(prompt, torch.arange(18, device=model.device), cache, bt, 0, last_only=True)
    toks = [int(prompt[-1])] + draft.tolist()
    for j, tok in enumerate(toks):
        pos = 18 + j
        ref.append(model.prefill_logits(
            torch.tensor([tok], device=model.device), torch.tensor([pos], device=model.device),
            cache, bt, pos, last_only=True)[0].float())
    ref = torch.stack(ref)

    cache2 = KVCache(cfg, 64, 16, model.device)
    bt2 = cache2.block_table_tensor([cache2.allocator.allocate() for _ in range(4)], 4)
    model.prefill_logits(prompt, torch.arange(18, device=model.device), cache2, bt2, 0, last_only=True)
    tdim = 16
    ids = torch.zeros(tdim, dtype=torch.long, device=model.device)
    pos_t = torch.zeros(tdim, dtype=torch.long, device=model.device)
    slot_t = torch.zeros(tdim, dtype=torch.long, device=model.device)
    ids[:K] = torch.tensor(toks, device=model.device)
    pos_t[:K] = torch.arange(18, 18 + K, device=model.device)
    slot_t[:K] = cache2.slots_of(bt2, 18, 18 + K)
    bt_row = torch.zeros((1, 4), dtype=torch.int32, device=model.device)
    bt_row[0] = bt2
    qs = torch.tensor([18], dtype=torch.int32, device=model.device)
    sl = torch.tensor([18 + K], dtype=torch.int32, device=model.device)
    bufs = model.get_buffers(tdim)
    bufs.pos[:K] = torch.arange(18, 18 + K, device=model.device)
    lg = model.verify_logits(ids, bufs.pos, slot_t, cache2, bt_row, qs, sl, tdim=tdim, n_real=K)
    for j in range(K):
        cos = torch.nn.functional.cosine_similarity(ref[j], lg[j].float(), dim=0).item()
        assert cos > 0.999, (j, cos)


@check("engine.speculative_single")
def test_speculative_single() -> None:
    """Speculation must not change the greedy output of a single sequence."""
    from kaze.engine import Engine, SamplingParams
    model = _load_model("int8")
    cycle = [11, 22, 33, 44, 55, 66]
    prompt = list(cycle) * 3 + [11, 22, 33]
    out = []
    steps = []
    for spec in (False, True):
        eng = Engine(model, max_batch=4, use_cuda_graph=False, speculative=spec, spec_k=5,
                     verbose=False, num_blocks=128, spec_accept_ema=4.0)
        seq = eng.add_request(prompt, SamplingParams(max_tokens=10, temperature=0.0))
        while seq.state != 3:
            eng.step()
        out.append(list(seq.output))
        steps.append(eng.stats["spec_steps"])
    assert out[0] == out[1], out
    assert steps[1] > 0, "single-sequence speculation never ran"


@check("moe.weight_bytes")
def test_moe_weight_bytes() -> None:
    """bookkeeping must not assume a dense MLP (MoE keeps experts stacked)"""
    model = _moe_model()
    if model is None:
        return
    n = model.weight_bytes()
    assert n > 0, n
    assert n < 4 * 1024 ** 3, n



@check("tile-major (int4 の並べ替え)")
def _() -> None:
    """tile-major の複製を作っても gemv_swiglu の出力が変わらないこと。"""
    import torch

    from kaze.kernels import quant_linear as K
    from kaze.quant import QuantLinear

    torch.manual_seed(7)
    n, k = 4096, 1024
    w = (torch.randn(n, k, device="cuda") * 0.02).half()
    x = (torch.randn(16, k, device="cuda") * 0.5).half()
    lin = QuantLinear.from_float(w, None, kind="int4", group=128)
    base = K.gemv_swiglu(x, lin, 16, n // 2).clone()
    lin.make_tile_major_copy()
    tm = K.gemv_swiglu(x, lin, 16, n // 2)
    cos = torch.nn.functional.cosine_similarity(
        base.float().flatten(), tm.float().flatten(), 0).item()
    assert cos > 0.999, f"tile-major の出力が一致しません (cos={cos:.6f})"



if __name__ == "__main__":
    sys.exit(main())


@check("moe.weight_bytes")
def test_moe_weight_bytes() -> None:
    """bookkeeping must not assume a dense MLP (MoE keeps experts stacked)"""
    model = _moe_model()
    if model is None:
        return
    n = model.weight_bytes()
    assert n > 0, n
    assert n < 4 * 1024 ** 3, n