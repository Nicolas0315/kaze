"""KL divergence vs an fp16 reference over several context lengths (multi-position)."""
import sys
sys.path.insert(0, ".")
import torch
from transformers import AutoTokenizer
from kaze.loader import load_model
from kaze.cache import KVCache

MODEL = sys.argv[1]
KIND = sys.argv[2] if len(sys.argv) > 2 else "int4"
GROUP = int(sys.argv[3]) if len(sys.argv) > 3 else 128
TEXT = ("The history of computing begins with mechanical calculators and continues "
        "through punched cards, relays, vacuum tubes and transistors. Each generation "
        "made the machines smaller, faster and cheaper, and each one changed what "
        "people could do with them. ") * 40
tok = AutoTokenizer.from_pretrained(MODEL)
ids_all = tok.encode(TEXT)
NCTX = int(sys.argv[4]) if len(sys.argv) > 4 else 8
LENS = [64 + 32 * i for i in range(max(NCTX, 1))]


HEAD = sys.argv[5] if len(sys.argv) > 5 else None


def collect(kind, group, head=None):
    m = load_model(MODEL, kind=kind, group=group, lm_head_kind=head or kind, verbose=False)
    dev = m.device
    out = []
    for L in LENS:
        # only as many blocks as this context needs: allocating the whole pool per context
        # exhausts 8 GB on 3B (the harness keeps two models resident)
        nblk = (L + 15) // 16 + 2
        cache = KVCache(m.cfg, nblk, 16, dev)
        blocks = [cache.allocator.allocate() for _ in range(nblk)]
        bt = cache.block_table_tensor(blocks, len(blocks))
        ids = torch.tensor(ids_all[:L], device=dev)
        with torch.no_grad():
            lg = m.prefill_logits(ids, torch.arange(L, device=dev), cache, bt, 0)
        out.append(lg[-1].float().cpu())
        del cache, blocks, bt, ids, lg
        torch.cuda.empty_cache()
    del m
    torch.cuda.empty_cache()
    return out


import os
# (a) default: fp16 reference in-process; (b) KL_DUMP=path: only save logits (for models
# too big to keep two copies resident, e.g. 3B on 8 GB); (c) KL_LOAD=path: compare
# against saved logits.
DUMP = os.environ.get("KL_DUMP")
LOAD = os.environ.get("KL_LOAD")
if DUMP:
    torch.save(collect(KIND, GROUP, HEAD), DUMP)
    print("dumped %d contexts to %s" % (len(LENS), DUMP))
    raise SystemExit(0)
if LOAD:
    ref = torch.load(LOAD, weights_only=False)
    test = collect(KIND, GROUP, HEAD)
    REF_LABEL = os.path.basename(LOAD)
else:
    REF_KIND = os.environ.get("KL_REF_KIND", "fp16")
    REF_GROUP = int(os.environ.get("KL_REF_GROUP", "128"))
    ref = collect(REF_KIND, REF_GROUP)
    test = collect(KIND, GROUP, HEAD)
    REF_LABEL = REF_KIND
kls, hits, bad = [], 0, 0
for r, t in zip(ref, test):
    if not torch.isfinite(t).all():
        bad += 1
        kls.append(float("nan"))
        continue
    lr, lt = torch.log_softmax(r, -1), torch.log_softmax(t, -1)
    kls.append((lr.exp() * (lr - lt)).sum().item())
    hits += int(r.argmax() == t.argmax())
finite = [k for k in kls if k == k]
print("%s %s group=%d vs %s: KL %.4f (mean of %d finite)  top-1 %d/%d  non-finite %d" % (
      MODEL.split("/")[-1], KIND, GROUP, REF_LABEL, sum(finite) / max(len(finite), 1),
      len(finite), hits, len(kls), bad))
print("  per-context:", " ".join("%.4f" % k for k in kls))
