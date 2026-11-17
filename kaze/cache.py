"""Paged KV cache: block allocator, block tables, prefix sharing.

Layout: [num_layers, num_blocks, block_size, kv_heads, head_dim] per K and V,
so a token slot is block_id * block_size + offset, and a layer slice is a
contiguous tensor the decode kernel can index directly.

Prefix caching hashes full blocks (chained with the parent hash) and shares the
same physical block between sequences with a refcount, like vLLM's APC.
"""
from __future__ import annotations

from collections import OrderedDict

import torch

from kaze.config import ModelConfig


class BlockAllocator:
    def __init__(self, num_blocks: int, block_size: int):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.free: list[int] = list(range(num_blocks - 1, -1, -1))
        self.ref: list[int] = [0] * num_blocks
        self.hash_to_block: dict[int, int] = {}
        # reverse index: release() needs "which hash does this block carry" once per
        # block, and scanning hash_to_block made freeing a 64K sequence O(blocks^2)
        # (measured 170 ms of engine stall at every request completion)
        self.block_to_hash: dict[int, int] = {}
        self.evictable: OrderedDict[int, int] = OrderedDict()   # block -> hash (ref == 0)

    @property
    def used(self) -> int:
        return self.num_blocks - len(self.free)

    def allocate(self) -> int:
        if not self.free:
            self.evict()
        if not self.free:
            raise MemoryError("KV cache exhausted")
        b = self.free.pop()
        self.ref[b] = 1
        return b

    def retain(self, block: int) -> None:
        self.ref[block] += 1
        self.evictable.pop(block, None)

    def release(self, block: int) -> None:
        self.ref[block] -= 1
        if self.ref[block] <= 0:
            self.ref[block] = 0
            h = self._block_hash(block)
            if h is not None:
                self.evictable[block] = h
            else:
                self.free.append(block)

    def _block_hash(self, block: int) -> int | None:
        return self.block_to_hash.get(block)

    def evict(self, need: int = 1) -> None:
        while len(self.free) < need and self.evictable:
            block, h = self.evictable.popitem(last=False)
            if self.hash_to_block.get(h) == block:
                del self.hash_to_block[h]
            self.block_to_hash.pop(block, None)
            self.free.append(block)

    def register_hash(self, h: int, block: int) -> None:
        old = self.block_to_hash.get(block)
        if old is not None and old != h and self.hash_to_block.get(old) == block:
            self.hash_to_block.pop(old, None)
        self.hash_to_block[h] = block
        self.block_to_hash[block] = h

    def lookup(self, h: int) -> int | None:
        return self.hash_to_block.get(h)


class KVCache:
    def __init__(self, cfg: ModelConfig, num_blocks: int, block_size: int, device,
                 dtype: torch.dtype = torch.float16,
                 dtypes: tuple[torch.dtype, torch.dtype] | None = None):
        self.cfg = cfg
        self.block_size = block_size
        self.device = device
        self.allocator = BlockAllocator(num_blocks, block_size)
        shape = (cfg.num_hidden_layers, num_blocks, block_size,
                 cfg.num_key_value_heads, cfg.head_dim)
        kd, vd = dtypes if dtypes is not None else (dtype, dtype)
        self.k = torch.zeros(shape, dtype=kd, device=device)
        self.v = torch.zeros(shape, dtype=vd, device=device)
        self.k_flat = self.k.view(cfg.num_hidden_layers, -1, cfg.kv_size)
        self.v_flat = self.v.view(cfg.num_hidden_layers, -1, cfg.kv_size)
        # int8 cache: scales exist for whichever half is int8 (lets K stay fp16 while
        # V is quantised and vice versa -- the two halves have very different error
        # sensitivity)
        self.q8k = kd == torch.int8
        self.q8v = vd == torch.int8
        self.q8 = self.q8k or self.q8v
        # one scale per 32 lanes of a head (q8_0 style), at least one
        self.n_scale = max(1, cfg.head_dim // 32)
        self.scale_shape = (cfg.num_hidden_layers, num_blocks * block_size,
                            cfg.num_key_value_heads, self.n_scale)
        self.ks = (torch.zeros(self.scale_shape, dtype=torch.float16, device=device)
                   if self.q8k else None)
        self.vs = (torch.zeros(self.scale_shape, dtype=torch.float16, device=device)
                   if self.q8v else None)
        # micro-block representatives for block-sparse decode attention
        self.mb = 4
        self.n_micro = num_blocks * block_size // self.mb
        # one mean key per 4-token micro-block (two representatives measured worse).
        # Only the sparse path reads it, and it costs ~4 bytes per cached token, so it
        # is allocated on first use (a 7B model on an 8GB card could not afford it).
        self.repr: torch.Tensor | None = None
        self._repr_dtype = torch.float16 if dtype == torch.int8 else dtype

    def ensure_repr(self) -> torch.Tensor:
        if self.repr is None:
            self.repr = torch.zeros(
                (self.cfg.num_hidden_layers, self.n_micro, 1, self.cfg.num_key_value_heads,
                 self.cfg.head_dim), dtype=self._repr_dtype, device=self.device)
        return self.repr

    def scales(self, layer: int):
        """(k_scale, v_scale) slices for one layer, either may be None."""
        ks = self.ks[layer] if self.q8k else None
        vs = self.vs[layer] if self.q8v else None
        return ks, vs

    @property
    def num_blocks(self) -> int:
        return self.allocator.num_blocks

    @property
    def free_blocks(self) -> int:
        return len(self.allocator.free)

    @property
    def used_blocks(self) -> int:
        return self.allocator.used

    def bytes_per_block(self) -> int:
        # size from the actual dtypes: fp8v keeps K in fp16 and V in fp8, so the same
        # memory holds 25% more tokens than an fp16 pool.  The engine derives the block
        # count from this number, so hardcoding 2 bytes per element left that capacity
        # unused (an external review measured "fp8v does not raise the KV token count,
        # VRAM is just left over").
        kb = self.k.element_size()
        vb = self.v.element_size()
        return (self.cfg.num_hidden_layers * self.block_size * self.cfg.kv_size
                * (kb + vb))

    # --------------------------------------------------------------- per seq
    def slots(self, blocks: list[int], start: int, end: int) -> torch.Tensor:
        idx = torch.arange(start, end, device=self.device, dtype=torch.int64)
        table = torch.tensor(blocks, device=self.device, dtype=torch.int64)
        return table[idx // self.block_size] * self.block_size + (idx % self.block_size)

    def slots_of(self, block_table: torch.Tensor, start: int, end: int) -> torch.Tensor:
        idx = torch.arange(start, end, device=self.device, dtype=torch.int64)
        return (block_table[idx // self.block_size].to(torch.int64) * self.block_size
                + (idx % self.block_size))

    def block_table_tensor(self, blocks: list[int], max_blocks: int) -> torch.Tensor:
        t = torch.zeros(max_blocks, dtype=torch.int32, device=self.device)
        n = min(len(blocks), max_blocks)
        if n:
            t[:n] = torch.tensor(blocks[:n], dtype=torch.int32, device=self.device)
        return t


    # ------------------------------------------------------- sparse attention
    def update_repr(self, layer: int, slots: torch.Tensor) -> None:
        """Refresh the mean key of every 4-token micro-block we just wrote."""
        mb = self.mb
        idx = torch.unique(slots // mb)
        src = self.k_flat[layer].view(self.n_micro, mb, self.cfg.kv_size)[idx]
        mean = src.mean(1).view(-1, 1, self.cfg.num_key_value_heads, self.cfg.head_dim)
        self.repr[layer].index_copy_(0, idx, mean)


def blocks_for(num_tokens: int, block_size: int) -> int:
    return (num_tokens + block_size - 1) // block_size
