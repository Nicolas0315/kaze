"""Backend detection and the per-vendor policy the engine follows.

The kernels are ordinary Triton, so they compile for whatever device PyTorch is bound
to.  What is not portable is the *tuning*: the tile tables were swept on an RTX 3070
(sm_86, 448 GB/s) and CUDA Graph capture is not equally mature on every stack.  This
module keeps that knowledge in one place instead of scattering `torch.cuda` checks.

`KAZE_BACKEND=rocm|cuda|cpu` forces the policy (not the device), so the ROCm policy can
be exercised end to end on a CUDA box.  `KAZE_GRAPHS=0|1` still overrides the graph part
of it, except on the CPU reference path where capture cannot work at all.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import torch


VENDORS = ("cuda", "rocm", "cpu")

_FLOAT8_CACHE_DTYPES = ("fp8", "fp8v")


@dataclass(frozen=True)
class Backend:
    vendor: str          # "cuda" | "rocm" | "cpu"
    device_name: str
    capability: tuple[int, int]
    graphs_ok: bool      # whether CUDA Graph capture is trusted on this stack
    fp8_ok: bool         # whether float8 KV dtypes are usable
    tuning_note: str


def _env_flag(name: str) -> bool | None:
    raw = os.environ.get(name)
    if raw is None:
        return None
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _env_backend() -> str | None:
    """KAZE_BACKEND as a vendor name, or None when unset/empty."""
    raw = os.environ.get("KAZE_BACKEND")
    if raw is None:
        return None
    value = raw.strip().lower()
    if not value:
        return None
    if value not in VENDORS:
        raise ValueError(f"KAZE_BACKEND must be one of {', '.join(VENDORS)}; got {raw!r}")
    return value


def real_vendor(device: str | torch.device | None = None) -> str:
    """The vendor PyTorch is actually bound to, ignoring KAZE_BACKEND."""
    dev = torch.device(device or "cuda")
    if dev.type != "cuda":
        return "cpu"
    return "rocm" if getattr(torch.version, "hip", None) else "cuda"


def _probe(dev: torch.device) -> tuple[str, tuple[int, int]]:
    """Device name and capability; degrade instead of failing before init."""
    if dev.type != "cuda":
        return "cpu", (0, 0)
    try:
        idx = dev.index or 0
        name = torch.cuda.get_device_name(idx)
        cap = torch.cuda.get_device_capability(idx)
    except Exception:  # device listed but not initialised yet
        return "unknown", (0, 0)
    return name, tuple(cap)


def detect_device(device: str | torch.device | None = None) -> Backend:
    """Describe the current backend and what it can be trusted to do."""
    dev = torch.device(device or "cuda")
    detected = real_vendor(dev)
    forced = _env_backend()
    vendor = forced or detected
    note_suffix = ""
    if forced is not None and forced != detected:
        note_suffix = f" Policy forced by KAZE_BACKEND={forced} (detected {detected})."

    if vendor == "cpu":
        # No capture and no float8 whatever the operator asked for: this path is the
        # reference implementation, and hipGraph/CUDAGraph both need a GPU stream.
        return Backend("cpu", "cpu" if dev.type != "cuda" else _probe(dev)[0], (0, 0),
                       False, False, "CPU reference path; kernels do not run here"
                       + note_suffix)

    name, cap = _probe(dev)
    if vendor == "rocm":
        # hipGraph exists, but capture through PyTorch's CUDAGraph wrapper is not reliable
        # across ROCm versions (allocator pools, stream capture on RDNA vs CDNA).  Default
        # to eager execution and let the operator opt in with KAZE_GRAPHS=1.
        graphs_ok = False
        # float8_e5m2 needs gfx94x-class hardware; the KV path stores/loads it directly.
        fp8_ok = cap[0] >= 9
        note = ("ROCm: kernels compile for this device, but the tile tables were tuned on "
                "sm_86. Expect lower throughput until retuned; CUDA Graph is off unless "
                "KAZE_GRAPHS=1 is set.")
    else:
        graphs_ok = cap[0] >= 7
        fp8_ok = True
        note = ("CUDA: tuned for sm_86 (RTX 3070, 448 GB/s). Other Ampere/Ada parts run "
                "but the tiles are not swept for their SM count or bandwidth.")

    override = _env_flag("KAZE_GRAPHS")
    if override is not None:
        graphs_ok = override
    return Backend(vendor, name, tuple(cap), graphs_ok, fp8_ok, note + note_suffix)


def cache_dtype_policy(cache_dtype: str, backend: Backend) -> str:
    """Downgrade float8 KV modes to fp16 when the backend cannot run them.

    8-bit KV is a memory saving, not a correctness feature: a backend without float8
    support would fail at the first cache write, so resolve it once here instead.
    """
    if cache_dtype in _FLOAT8_CACHE_DTYPES and not backend.fp8_ok:
        return "fp16"
    return cache_dtype


def describe(b: Backend) -> str:
    arch = f" sm_{b.capability[0]}{b.capability[1]}" if b.capability != (0, 0) else ""
    return (f"backend={b.vendor} device={b.device_name}{arch} "
            f"graphs={'on' if b.graphs_ok else 'off'} fp8={'ok' if b.fp8_ok else 'unsupported'}")
