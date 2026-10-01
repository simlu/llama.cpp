# Prefill port plan: GGML_CUDA_REGISTER_HOST + GGML_SCHED_PREFETCH_EXPERTS

Branch: leloch-v3 (origin/master simlu/llama.cpp @ 19e28a277, +31 moe-cache commits)
Source fork: https://github.com/thecodacus/llama.cpp, branch `perf`
Date: 2026-10-01
Scope: review + plan only. No code was changed.

## 1. Bottom line

Both features are small, self-contained, opt-in fork commits that target exactly the
scenario this branch is built for: a large MoE model whose routed experts are kept in
system RAM via `--n-cpu-moe` (`-ncmoe`), streaming weights to the GPU during prefill.

- `GGML_CUDA_REGISTER_HOST=1` (pin mmap'd CPU weights): the backend function and its
  env gate are **already in this branch** (upstream, not fork-specific). Only the
  **calling site** is missing. The fork's entire contribution is a 3-file wiring change.
  Easy port.
- `GGML_SCHED_PREFETCH_EXPERTS=1` (overlap expert uploads with compute): **not present**
  in the branch. It is a 1-file scheduler change (2 commits). The fork wrote it against
  a base that already had the branch's MoE expert-copy path, so it is a clean delta.

All required ggml APIs exist in the branch. CUDA reports `caps.async=true`,
`caps.events=true` (unless `GGML_CUDA_NO_PEER_COPY`), so prefetch will engage.

## 2. Source of truth (fork commits, all ancestors of perf tip)

| SHA | Subject | Files |
| --- | --- | --- |
| `20f5994bfeb91d24da328077c4b6095998cc9888` | llama : pin mmap-backed CPU weights for faster H2D uploads | src/llama-mmap.cpp, src/llama-mmap.h, src/llama-model-loader.cpp |
| `1163cb34939fe4a9cb07aec034c5954144497ae9` | ggml : overlap offloaded expert weight uploads with compute | ggml/src/ggml-backend.cpp |
| `5f83fbbe7c668c59912a1fe09e86a0ef580406c4` | ggml : size prefetch slots per layer and fix fallback use-after-free | ggml/src/ggml-backend.cpp |

Commit `20f5994bf` is the REGISTER_HOST wiring. `1163cb349` + `5f83fbbe7` are the
prefetch feature (the second fixes a use-after-free and bumps the default slot count
from 2 to 3; both are required). The fork's README also references a separate
`fable5/host-register` branch, but the `perf` tip already contains both features.

## 3. Feature A: GGML_CUDA_REGISTER_HOST

### 3.1 What it is

When experts are offloaded to CPU (`--n-cpu-moe`), the model weights live in mmap'd
file pages (pageable memory). `cudaMemcpyAsync` from pageable memory goes through the
driver's bounce buffer and is slow. `cudaHostRegister()` page-locks those pages so the
host->device copies go straight over DMA. The fork measured 1144 -> 1385 t/s prefill on
an RTX 3060 for Qwen3.6-35B-A3B at pp2048.

### 3.2 Current branch state (important)

`ggml_backend_cuda_register_host_buffer()` and `ggml_backend_cuda_unregister_host_buffer()`
**already exist** in this branch at `ggml/src/ggml-cuda/ggml-cuda.cu:4939` and `:4962`,
with the `getenv("GGML_CUDA_REGISTER_HOST")` gate. They are exposed as support functions
(`ggml-cuda.cu:5800-5804`) and declared in `ggml/include/ggml-cuda.h:40-41`. This code is
from **upstream** (commit `47373271f` HIP enable + `c3d6af7cd` CUDART fix), NOT from the
fork. Verified against upstream raw `master` - the function and gate are identical.

The fork's README describes this as a fork optimization, but the fork's actual addition
is the **call site**: nothing in this branch ever invokes `register_host_buffer`. The
fork commit `20f5994bf` "wires the existing GGML_CUDA_REGISTER_HOST path back up".

So for this feature, do NOT touch ggml-cuda.cu. Port only the 3-file call site.

### 3.3 Changes to port (from `20f5994bf`)

**src/llama-mmap.h** (after the existing `unmap_fragment` decl, line 56):
- Add public method:
  `size_t register_host(size_t first, size_t last, bool (*reg_fn)(void *, size_t), void (*unreg_fn)(void *));`
- Add private members: `void * host_reg_addr = nullptr;` and
  `void (*host_unreg_fn)(void *) = nullptr;`
- Note: the branch's `llama_mmap` ctor already has the extra `const ranges & lazy_ranges = {}`
  param (upstream evolution). Keep it; the fork's members/method merge in cleanly.

**src/llama-mmap.cpp**:
- Change `llama_mmap::~llama_mmap() = default;` (line 668) to a custom destructor that
  calls `host_unreg_fn(host_reg_addr)` when `host_reg_addr && host_unreg_fn` (unpin before
  the impl destructor unmaps).
- Add `llama_mmap::register_host(...)` after `unmap_fragment` (line 673):
  - Guard `#ifdef _POSIX_MAPPED_FILES` (present in the branch, line 466).
  - Expand `[first,last)` outward to page boundaries using `sysconf(_SC_PAGESIZE)`.
  - `reg_addr = (uint8_t *) pimpl->addr + first;` then `reg_fn(reg_addr, last - first)`.
  - On success store `host_reg_addr` / `host_unreg_fn`, return `last - first`; else 0.
  - Non-POSIX: return 0.
- The branch's `impl` already has `void * addr; size_t size;` (lines 662-663), so
  `pimpl->addr` is accessible.

**src/llama-model-loader.cpp** (`load_all_data`, at line 1777 `if (size_done >= size_data)`):
- Inside the `if (use_mmap)` block (line 1779), before the unmap loop, look up the
  backend proc addresses:
  - Iterate `ggml_backend_dev_count()` / `ggml_backend_dev_get(i)`, get
    `ggml_backend_dev_backend_reg(dev)`, then
    `ggml_backend_reg_get_proc_address(reg, "ggml_backend_register_host_buffer")` /
    `"ggml_backend_unregister_host_buffer"`. Stop at the first dev that provides `reg_fn`.
- In the per-mapping loop, after the existing `unmap_fragment` calls, add:
  `if (mmap_used.second > mmap_used.first) { n = mapping->register_host(mmap_used.first, mmap_used.second, reg_fn, unreg_fn); if (n > 0) LLAMA_LOG_INFO(...); }`
- The branch's loop already computes `mmap_used.first`/`.second` (lines 1783-1785). The
  branch's structure unmaps BOTH the head `(0, mmap_used.first)` and the tail
  `(mmap_used.second, size)`; the fork's base only unmapped the tail. The pinning call
  goes right after those, using the same `mmap_used` range. `mmap_used.first`/`.second`
  already span the offloaded weights retained in host memory (set at lines 1666-1668).
- `LLAMA_LOG_INFO` is already used in this file. `ggml_backend_dev_count()` etc. are
  reachable via the already-transitive `ggml-backend.h` (the file already calls
  `ggml_backend_dev_host_buffer_type`, `ggml_backend_buffer_is_host`).

### 3.4 Scope check

`register_host` pins the mmap pages backing ALL weights that stay in system RAM (the
`mmaps_used` ranges), which is exactly the `--n-cpu-moe` working set. It is opt-in via
the env var (gate is inside the CUDA function). No output change; the pages are
unregistered in the destructor. This is the smaller, lower-risk of the two features.

## 4. Feature B: GGML_SCHED_PREFETCH_EXPERTS

### 4.1 What it is

In the split scheduler's input-copy path, mainline reads back the routing-id tensor,
synchronizes, then copies only the used experts per layer. At large prefill batch that
readback buys nothing (virtually every expert is used) while forcing a full device sync
per expert tensor (3x per MoE layer). Above a batch threshold, this feature skips the ID
readback and uploads the FULL expert tensor through a second backend instance on the same
device (its own stream), alternating among event-ordered staging slots so upload for
tensor N+1 overlaps compute of tensor N. Fork measured 1383 -> 1663 -> 1880 t/s (after the
slot-count fix) vs mainline 1143 t/s, token-identical output, decode path unaffected.

### 4.2 Key integration finding

The branch's `compute_splits` (`ggml/src/ggml-backend.cpp:1727`) already contains the
MoE expert-selective copy path (`prev_ids_tensor`, `ids`, `used_ids` bitset, `copy_experts`
lambda) that was ported as part of the moe-cache series. I checked the parent of the fork's
prefetch commit (`1163cb349^`): it has the SAME `copy_experts`/`used_ids` structure and
does NOT have `cpu_async` (`GGML_SCHED_ASYNC_CPU`). This exactly matches the branch. So the
prefetch commits are a clean delta over the branch's current state - the prefetch check
sits at the top of the input-copy `else` branch and takes priority when the batch is large
(`ids->ne[0]*ids->ne[1] >= 2*n_expert`), otherwise it falls through to the existing
expert-copy path. The branch's `moe_cache_scope` RAII (lines 1731-1750) and the
`split->n_inputs == 0 && prev_backend_id` sync block (1765-1771) are branch-only additions
the fork base did not have, so a cherry-pick will need a light 3-way merge there, but the
prefetch logic itself lands unchanged.

### 4.3 Changes to port (from `1163cb349` + `5f83fbbe7`), all in ggml/src/ggml-backend.cpp

**A. Macro** - after `GGML_SCHED_MAX_COPIES` (line 836):
`#ifndef GGML_SCHED_MAX_PREFETCH_SLOTS / #define GGML_SCHED_MAX_PREFETCH_SLOTS 8 / #endif`

**B. `struct ggml_backend_sched`** - after `bool op_offload;` (line 897), add:
- `bool prefetch_experts;`
- `ggml_backend_t prefetch_backend;`
- `int prefetch_n_slots;`
- `ggml_backend_buffer_t prefetch_slots[GGML_SCHED_MAX_PREFETCH_SLOTS];`
- `ggml_backend_event_t prefetch_ready[...]`, `prefetch_free[...]`, `bool prefetch_used[...]`
- `int prefetch_cur;`

**C. Three static helpers** before `ggml_backend_sched_compute_splits` (insert near
line 1726, after `ggml_backend_sched_alloc_splits`):
- `ggml_backend_sched_prefetch_disable(sched, split_backend)` - clears `prefetch_experts`,
  syncs both backends, frees all slots, resets `prefetch_used`.
- `ggml_backend_sched_prefetch_max_size(sched)` - scan splits for MUL_MAT_ID with a
  host WEIGHTS buffer input, return the max `ggml_nbytes(input)`.
- `ggml_backend_sched_prefetch_init(sched, split_backend, size)`:
  - If `prefetch_backend == NULL`: check `props.caps.async` && `props.caps.events`
    (`ggml_backend_dev_get_props`); create the second backend via
    `ggml_backend_dev_init(dev, NULL)`; create `prefetch_n_slots` ready/free event pairs.
    Any failure disables prefetch and returns false.
  - `size = std::max(size, ggml_backend_sched_prefetch_max_size(sched));`
  - For each slot: if NULL or smaller than `size`, allocate `new_buf` first (so a failure
    leaves the old slot intact), then free the old. On OOM: if `i >= 2` and slot[0] fits,
    reduce `prefetch_n_slots = i` and return true (degrade); else disable and return false.

**D. `ggml_backend_sched_compute_splits`** (the only behavioral change in compute):
- At the top of the split loop (after line 1761 `split_backend`), add locals:
  `int split_prefetch_slot = -1; ggml_tensor * prefetch_input_cpy = NULL;
   ggml_backend_buffer_t prefetch_saved_buffer = NULL; void * prefetch_saved_data = NULL;`
- In the input-copy `else` branch, right after the event_wait/sync (after line 1793,
  before the `ggml_tensor * node = split->graph.nodes[0];` line 1796), insert the prefetch
  block:
  - Condition: `sched->prefetch_experts && !sched->callback_eval && split_prefetch_slot == -1
    && split->graph.n_nodes > 0`
  - `node = split->graph.nodes[0];` and require
    `usage == WEIGHTS && is_host && node->op == GGML_OP_MUL_MAT_ID && node->src[0] == input_cpy`
  - `ids = node->src[2]; n_expert = input->ne[2];`
  - If `ids->ne[0]*ids->ne[1] >= 2*n_expert && ggml_backend_sched_prefetch_init(...)`:
    - `slot = prefetch_cur; prefetch_cur = (prefetch_cur + 1) % prefetch_n_slots;`
    - if `prefetch_used[slot]`: `ggml_backend_event_wait(prefetch_backend, prefetch_free[slot])`
    - save `input_cpy->buffer`/`data`, repoint to `prefetch_slots[slot]`, call
      `ggml_backend_tensor_set_async(prefetch_backend, input_cpy, input->data, 0, nbytes)`,
      `ggml_backend_event_record(prefetch_ready[slot], prefetch_backend)`,
      `ggml_backend_event_wait(split_backend, prefetch_ready[slot])`.
    - `split_prefetch_slot = slot;` then `continue` (skips the expert-copy path for this input).
- In the `!sched->callback_eval` compute block, right after
  `ggml_backend_graph_compute_async(...)` (line 1901), BEFORE the `if (ec != SUCCESS)` check:
  - If `split_prefetch_slot != -1`: record `prefetch_free[split_prefetch_slot]` on
    `split_backend`, set `prefetch_used[split_prefetch_slot] = true`, and restore
    `prefetch_input_cpy->buffer`/`data` to the saved values. (Restoring right after launch
    is what fixes the use-after-free: the kernels have already captured the slot address.)

**E. `ggml_backend_sched_new`** - after `sched->op_offload = op_offload;` (line 2023), add:
- Read `getenv("GGML_SCHED_PREFETCH_EXPERTS")`; `prefetch_n_slots = atoi(...)` (0 if unset).
- `sched->prefetch_experts = op_offload && prefetch_n_slots > 0;`
- `sched->prefetch_n_slots = prefetch_n_slots <= 1 ? 3 : std::min(prefetch_n_slots, GGML_SCHED_MAX_PREFETCH_SLOTS);`

**F. `ggml_backend_sched_free`** - in the cleanup (after the moe-cache destroy, line 2067),
add prefetch teardown: sync `prefetch_backend`, free all ready/free events and slots, free
`prefetch_backend`. Loop over `GGML_SCHED_MAX_PREFETCH_SLOTS` (slot count may have shrunk).

**G. `ggml_backend_sched_synchronize`** - after the backends loop (line 2180), add
`if (sched->prefetch_backend) ggml_backend_synchronize(sched->prefetch_backend);`

## 5. Port strategy

The fork is already cloned at /tmp/thecodacus-llama. Recommended, matching the existing
moe-cache port style (which used `-x` provenance):

1. `git remote add thecodacus https://github.com/thecodacus/llama.cpp.git` and fetch
   `perf` (or cherry-pick directly from the /tmp clone via a local remote).
2. Cherry-pick with provenance, in this order:
   - `20f5994bf` (REGISTER_HOST wiring) - expect small conflicts in llama-mmap.cpp
     (branch ctor has `lazy_ranges`) and llama-model-loader.cpp (branch unmap loop shape).
   - `1163cb349` (prefetch v1) then `5f83fbbe7` (prefetch v2 fix) - expect conflicts in
     `ggml-backend.cpp` around the `moe_cache_scope`/`split->n_inputs == 0` additions and
     line drift; the prefetch hunks themselves apply with the context in section 4.3.
3. If a hunk fights, do a 3-way merge with `git merge-file` (branch as current,
   `merge-base` of the fork feature commit and the branch as base, fork HEAD as other),
   keeping branch semantics and preserving the branch's moe-cache code.
4. Do NOT bring in the fork's `cpu_async` / `GGML_SCHED_ASYNC_CPU` or the MoE expert-cache
   commits (already in the branch via the moe-cache port). Only the 3 SHAs in section 2.

Because the branch base already matches the prefetch base, an alternative to cherry-picking
is to hand-apply the section 4.3 hunks directly - the delta is self-contained.

## 6. Risks and interactions

- **Both features target the same `--n-cpu-moe` path.** REGISTER_HOST pins the CPU pages
  (DMA-friendly copies); PREFETCH overlaps the (now-DMA) uploads with compute. They compose:
  the fork measured the combined result (1880 t/s) with both env vars on. Port both.
- **moe-cache interaction.** The branch's moe-cache keeps hot experts VRAM-resident; the
  prefetch streams the cold CPU remainder. The fork shipped both together, so they coexist,
  but this combination is the least-tested path. Validate that the moe-cache's CUDA expert
  path and the scheduler prefetch do not double-handle the same tensor, and that the
  `GGML_OP_OFFLOAD_MIN_BATCH` threshold does not bypass the cache.
- **VRAM cost.** Prefetch allocates `prefetch_n_slots` (default 3, max 8) staging buffers,
  each sized to the largest offloaded expert tensor. On a small GPU this can OOM; the code
  degrades by shrinking the slot count (v2 fix), which is the desired fallback. Set
  `GGML_SCHED_PREFETCH_EXPERTS` to a smaller value to cap it.
- **`caps.events` gate.** On builds with `GGML_CUDA_NO_PEER_COPY`, `caps.events` is false
  and prefetch silently disables itself. This is by design; document that prefetch needs
  peer-copy-enabled CUDA.
- **No GPU on this machine.** The moe-cache report already flagged CUDA runtime as
  unverified here. Both new features are CUDA-runtime-only. This machine can verify
  compilation but NOT performance or correctness of the DMA/stream paths.
- **`register_host` proc-address lookup.** The fork looks up `ggml_backend_register_host_buffer`
  from the first backend that exposes it. On a CUDA build the CUDA backend wins; on a
  CPU-only build there is no implementation and the call returns 0 (no-op). Fine.

## 7. Verification plan

Compile (this machine, no GPU):
- CPU build: `cmake -B build-cpu -DGGML_CUDA=OFF -DLLAMA_BUILD_TESTS=ON` and build. Both
  features are CUDA-gated / CPU-noop, so a CPU build should pass unchanged.
- CUDA compile-only: `cmake -B build-cuda -DGGML_CUDA=ON -DLLAMA_BUILD_TESTS=ON`; confirm
  `ggml-backend.cpp`, `llama-mmap.cpp`, `llama-model-loader.cpp` recompile and all targets
  build (matching the moe-cache report's gate).
- `git diff --check` and ASCII check on the branch diff.

Runtime (needs a real GPU - follow the moe-cache report's hardware handoff):
- llama-bench, 3 arms at pp2048 with `-ncmoe <N> -ngl 99`:
  baseline (both off), REGISTER_HOST only, both on. Fork reference numbers:
  1143 -> 1385 (register) -> 1880 t/s (both) on RTX 3060 with Qwen3.6-35B-A3B.
- Verify token-identical output vs baseline (prefill and a decode pass).
- Test the prefetch degradation path on a small VRAM budget (force OOM, confirm slot
  shrink and no crash / no UAF).
- Confirm `--help` shows no new CLI flags (both are env-var only).

## 8. Rollback

Nothing is pushed and no PR exists. Each commit carries its `(cherry picked from commit ...)`
provenance line, so the port can be reverted per-feature (`git revert` the 3 SHAs, or
`git reset` the branch back to fd349cf6a). The fork's reference commits and the /tmp clone
remain available for re-inspection.
