# Port plan: GGML_CUDA_REGISTER_HOST + GGML_SCHED_PREFETCH_EXPERTS onto leloch-v4

Investigation only - no code was changed.

Reference fork: https://github.com/thecodacus/llama.cpp (local ref `thecodacus/perf`,
already fetched in /workspace).
Target branch: leloch-v4 (a3c79a143, MoE expert-cache v4 work on recent upstream master).

Both features target the same bottleneck: **prefill of MoE models whose routed experts
live in system RAM** (`--n-cpu-moe` / `--cpu-moe`). During prefill every token batch
touches (nearly) every expert, so the H2D weight streaming path dominates: mainline
ggml synchronizes per expert tensor to read back routing ids, then issues partial
expert copies on the compute stream, serializing upload and compute over a pageable
mmap buffer.

Fork-claimed results (RTX 3060 12GB, Qwen3.6-35B-A3B, `-ncmoe 26 -ub 2048`, pp2048):
baseline 1143 t/s -> host-register alone 1385 t/s -> + prefetch 1880 t/s (+64%),
decode unaffected, output token-identical. Numbers are from the fork README; treat as
unverified until we reproduce.

Relevant upstream commits on thecodacus/perf:

| SHA        | What |
| ---        | --- |
| 20f5994bf  | llama : pin mmap-backed CPU weights for faster H2D uploads (REGISTER_HOST wiring) |
| 1163cb349  | ggml : overlap offloaded expert weight uploads with compute (PREFETCH_EXPERTS v1, 2 slots) |
| 5f83fbbe7  | ggml : size prefetch slots per layer and fix fallback use-after-free (final shape, N slots) |

Note: `GGML_CUDA_REGISTER_HOST` itself is NOT new code - it is the upstream env gate
from slaren's `d0a71233f (cuda : disable host register by default, #6206)` (2024).
It already exists verbatim on leloch-v4 at `ggml/src/ggml-cuda/ggml-cuda.cu:4947`.
What the fork added is the *caller*: upstream registers nothing because nothing calls
the proc address anymore.

---

## Feature 1 - GGML_CUDA_REGISTER_HOST=1 (pin mmap weights)

### How it works upstream today

- `ggml_backend_cuda_register_host_buffer(buffer, size)` in ggml-cuda.cu:
  no-op returning false unless `GGML_CUDA_REGISTER_HOST` is set; when set, calls
  `cudaHostRegister(buf, size, cudaHostRegisterPortable | cudaHostRegisterReadOnly)`,
  clears the error and returns false on failure. `..._unregister_host_buffer` symmetric.
- Exported through the CUDA backend registry as
  `"ggml_backend_register_host_buffer"` / `"ggml_backend_unregister_host_buffer"`
  (ggml-cuda.cu:5819 on leloch-v4 - already present).
- On leloch-v4 **no caller exists** in src/ or ggml/ (only commented-out SYCL stubs).
  So the flag is dead: page-locking never happens for mmap'd weights.

### What thecodacus adds (commit 20f5994bf, +~50 lines, 3 files)

1. `src/llama-mmap.h` / `src/llama-mmap.cpp` - new method
   `size_t llama_mmap::register_host(first, last, reg_fn, unreg_fn)`:
   - guards: already registered / null fns / empty range -> 0
   - expands [first, last) outward to host page boundaries (`sysconf(_SC_PAGESIZE)`)
   - calls `reg_fn(addr + first, last - first)`; on success remembers
     `host_reg_addr` + `host_unreg_fn` members
   - `~llama_mmap()` unregisters before the impl destructor unmaps the pages.
2. `src/llama-model-loader.cpp` - in `load_all_data()`'s final cleanup block
   (`if (size_done >= size_data) { if (use_mmap) { ... } }`):
   - once, scans `ggml_backend_dev_count()` for the first backend registry that
     exposes `ggml_backend_register_host_buffer` (i.e. CUDA/MUSA/HIP)
   - after `unmap_fragment()` of the unused head/tail of each mapping, calls
     `mapping->register_host(mmap_used.first, mmap_used.second, reg_fn, unreg_fn)`
     and logs `pinned %.2f MiB ...` when > 0.

Net effect: exactly the byte-range of the GGUF mmap that backs CPU-resident weights
(what `mmaps_used` retained) is page-locked for the lifetime of the model. All H2D
copies from it (both the partial-expert copies and full loads) go over DMA from
pinned pages instead of the driver's pageable bounce path (~6-7 -> ~20 GB/s per the
fork).

Why it matters for correctness too: `cudaHostRegister` is applied to the *retained*
fragment only, after the head/tail `munmap`s, so it can never register a hole; the
destructor ordering (unregister before unmap) avoids the classic use-after-unmap.

### Port to leloch-v4

Trivial. Verified the anchor block still exists in the same shape on leloch-v4
(src/llama-model-loader.cpp:1776-1788 - `// check if this is the last call and do
final cleanup` / `// unmap offloaded tensors and metadata`), and
`src/llama-mmap.cpp:681` still has `llama_mmap::~llama_mmap() = default;`
(-> replace with the unregistering dtor). Upstream churn since the fork's base
(`32dd62ee6` direct-io mmap change, `81aeaeb74` GGUF alignment) touches adjacent
loading code but not these anchors' semantics.

Recommended: `git cherry-pick -x 20f5994bf`. Expect at most whitespace conflicts in
llama-model-loader.cpp. No changes needed in ggml-cuda.cu - the gated function is
byte-identical on leloch-v4.

Risk notes:
- Page-locking multi-GB requires sufficient `RLIMIT_MEMLOCK` / `CAP_IPC_LOCK`;
  on failure the fork code logs nothing extra and simply returns 0 (register_host
  bails, `reg_fn` failure is already logged at GGML_LOG_DEBUG in ggml-cuda.cu).
  Graceful. Worth adding our own one-line info if 0 bytes registered while
  `-ncmoe > 0` and the env var is set (optional).
- `cudaHostRegisterReadOnly` is correct here: weights are read-only from the
  process side after load.
- Leloch MoE-cache interplay: the v4 moe-cache copies experts CPU->GPU into cache
  pools from the same mmap-backed host buffers, so pinning speeds cache fills too.
  No address invalidation concern: moe_cache.invalidate hooks watch buffer frees;
  cudaHostRegister doesn't move addresses.

---

## Feature 2 - GGML_SCHED_PREFETCH_EXPERTS=1 (stream-ahead full-tensor upload)

### The problem it solves

In `ggml_backend_sched_compute_splits()` (present identically on both branches), for
each split input that is a host WEIGHTS tensor feeding `MUL_MAT_ID` (`node->src[0] ==
input_cpy`), mainline:

1. `ggml_backend_synchronize(input_backend)` and reads the routing `ids` tensor back
   to CPU (`tensor_get_async` + `synchronize`) - a full device sync;
2. builds a bitset of used experts, groups consecutive runs, issues
   `ggml_backend_tensor_set_async(split_backend, ...)` per run.

This stalls the compute stream between splits: sync, then copy, then compute, with
the copy serialized ahead of the kernel it feeds. 3x per MoE layer (gate/up/down).

At prefill-sized batches (tokens x n_experts_used >= 2 x n_expert) virtually every
expert is touched, so the ids readback buys nothing - just upload the *whole* tensor,
and do it on a *second* backend instance (own CUDA stream) one tensor ahead of compute
so upload N+1 overlaps compute N.

### How thecodacus implements it (final state after 1163cb349 + 5f83fbbe7, ~200 lines, ggml-backend.cpp only)

State in `struct ggml_backend_sched`:

```c
#define GGML_SCHED_MAX_PREFETCH_SLOTS 8   // file-local

bool prefetch_experts;                    // env-enabled && op_offload
ggml_backend_t prefetch_backend;          // 2nd instance on same device (2nd stream)
int prefetch_n_slots;                     // default 3 = gate/up/down of one layer;
                                          // env value >1 sets it directly, capped at 8
ggml_backend_buffer_t prefetch_slots[8];  // device staging, one max-tensor per slot
ggml_backend_event_t prefetch_ready[8];   // recorded on prefetch stream after upload
ggml_backend_event_t prefetch_free[8];    // recorded on compute stream after launch
bool prefetch_used[8];
int prefetch_cur;                         // ring cursor
```

Activation in `ggml_backend_sched_new()`:
`prefetch_experts = op_offload && atoi(getenv("GGML_SCHED_PREFETCH_EXPERTS")) > 0;`
(default 3 slots; value 1 -> 3, value N -> N capped at 8).

In `ggml_backend_sched_compute_splits()`, per split, inserted in the input-copy loop
*before* the existing ids-readback path (so it shadows it when eligible):

- Eligibility: `prefetch_experts && !callback_eval && no slot taken for this split
  yet && split->graph.n_nodes > 0`; input is host WEIGHTS buffer; first node is
  `MUL_MAT_ID` with `src[0] == input_cpy`; and
  `ids->ne[0] * ids->ne[1] >= 2 * n_expert` (dense-batch heuristic).
- `prefetch_init`: lazily creates the second backend via `ggml_backend_dev_init(dev,
  NULL)` (requires `props.caps.async && props.caps.events`), allocates slots sized to
  `max(ggml_nbytes(input), ggml_backend_sched_prefetch_max_size(sched))` - a scan over
  all splits' host-weight inputs feeding MUL_MAT_ID so slots are sized once per graph
  and never need to grow mid-eval.
  - allocates the new buffer **before** freeing the old (5f83fbbe7 fix);
  - on OOM: degrade to however many slots fit if >= 2 already fit, else
    `prefetch_disable()` (sync both backends, free slots) and fall through to the
    normal path.
- Slot rotation for the chosen input:
  1. `slot = prefetch_cur; prefetch_cur = (prefetch_cur+1) % n_slots`
  2. if slot was used before: prefetch stream `event_wait(free[slot])`
  3. **temporarily** repoint `input_cpy->buffer/data` at the slot
     (saving the originals in locals - the 5f83fbbe7 use-after-free fix: repoints are
     restored right after launch so graph reuse / fallback paths can never observe a
     dangling slot pointer);
  4. `ggml_backend_tensor_set_async(prefetch_backend, input_cpy, input->data, 0, nbytes)`
     - full tensor on stream B, overlapping compute already queued on stream A;
  5. `event_record(ready[slot], prefetch_backend)` then
     `event_wait(split_backend, ready[slot])` - compute stream blocks on the *copy*,
     not on the CPU;
  6. `continue` skips the ids readback entirely.
- After `ggml_backend_graph_compute_async(split_backend, &split->graph)` returns
  (kernels have captured the slot address at launch): record `free[slot]` on the
  compute stream, mark `prefetch_used[slot] = true`, restore `input_cpy->buffer/data`.
- Teardown: `ggml_backend_sched_free` syncs + frees events/slots/2nd backend (loops
  the full MAX array since n_slots may have been reduced);
  `ggml_backend_sched_synchronize` also syncs `prefetch_backend`.

Important scope facts:
- Only one slot per split (only the first eligible MUL_MAT_ID input per split takes
  the fast path; a split carries one expert tensor, so lookahead comes from the ring
  across the 3 tensors of consecutive splits).
- `!callback_eval` guard: the chunked callback path is never prefetched.
- The ids-partial-copy fallback stays fully intact; when prefetch is off, ineligible,
  or disabled mid-run, behavior is bit-identical to mainline.
- The fork ALSO carries `GGML_SCHED_ASYNC_CPU` (cpu_async) in ggml-backend.cpp - that
  is a separate feature woven through the same function's context lines. Do not port
  it; it is the main cherry-pick nuisance.
- Fork commit 08fa762f7 ("mul_mat_id sync predicate must account for the expert-pack
  split") belongs to their own hot/cold expert-pack work, not to these two flags.

---

## Divergence of leloch-v4 vs the fork's base (what the port must respect)

leloch-v4 rewrote big parts of the same neighborhoods:

1. `ggml/src/ggml-backend.cpp` carries the leloch MoE-cache v4 integration:
   - `struct ggml_moe_cache_api ggml_moe_cache` function table
     (ggml/src/ggml-backend-moe-cache.h),
   - `moe_cache_session` field on the sched + `session_create/destroy` in
     sched_new/free and `ggml_backend_sched_set_moe_cache()`,
   - `moe_cache_scope` RAII enter/leave wrapping the whole of
     `ggml_backend_sched_compute_splits()` (leloch ggml-backend.cpp:1727-1750),
   - `ggml_backend_moe_cache_invalidate_buffer` hooks on every buffer
     alloc/set/free path,
   - ids loop asserts `id >= 0` (their pack split encodes non-owned experts
     differently than the fork's `if (id < 0) continue`).
2. The fork's perf branch has its own older MoE-cache (`ac743f81f`, `0ac3d9b27`) plus
   cpu_async - these diverged lines are why a plain cherry-pick of 1163cb349/5f83fbbe7
   will conflict in `ggml_backend_sched` struct, sched_new, sched_free, synchronize,
   and the head of compute_splits.
3. Semantic overlap to decide on: leloch-v4's MoE cache already keeps *hot* experts
   in VRAM and trims the CPU-side MUL_MAT_ID work (skip dormant sessions, bounded
   admission). PREFETCH_EXPERTS targets the remaining *cold/host* weight upload
   stream. They are complementary, not competing:
   - prefetch only fires on host-WEIGHTS-buffer inputs whose first split node is a
     plain MUL_MAT_ID reading `src[0] == input_cpy`; leloch-cache'd nodes execute
     through the cache dispatch path, and dormant/forced modes bypass. Verify at
     port time that a cache-managed tensor never satisfies the eligibility check
     while a cache session is active for it - if it can, gate prefetch with
     `!sched->moe_cache_session` first, then relax behind a flag if benchmarks show
     they compose.
   - the dense-batch predicate `ids->ne[0]*ids->ne[1] >= 2*n_expert` already
     self-excludes decode (ids are ~n_experts_used x n_tokens), matching the fork's
     "decode unaffected" claim.

## Step-by-step port plan (for the follow-up implementation task)

Order matters: land the small one first, benchmark arms separately.

1. `git checkout -b prefill-optimizations leloch-v4` (or work directly on a scratch
   branch; nothing gets committed without user approval).
2. `git cherry-pick -x 20f5994bf` (host-register wiring).
   - Expect clean or trivial conflict in llama-model-loader.cpp final-cleanup block.
   - Keep the `Assisted-by`/provenance convention used by v3/v4 picks.
3. Apply the prefetch feature **manually** as a fresh commit (do not cherry-pick
   1163cb349 then 5f83fbbe7 - the intermediate state and cpu_async context make it
   worse; take only the post-5f83fbbe7 final shape from
   `git show thecodacus/perf:ggml/src/ggml-backend.cpp`):
   a. struct fields + `GGML_SCHED_MAX_PREFETCH_SLOTS` define, placed after
      `bool op_offload;` / before `int debug;` (skip the fork's `cpu_async` line);
   b. `prefetch_disable` / `prefetch_max_size` / `prefetch_init` helpers after
      `ggml_backend_sched_alloc_splits`;
   c. eligibility block inside the input-copy `else` branch, ahead of the
      `wait for the split backend` lines (leloch ggml-backend.cpp:~1789);
   d. slot-release + repoint-restore right after `ggml_backend_graph_compute_async`
      in the `!callback_eval` branch (mind leloch's moe_cache_scope - it wraps the
      function, no interaction beyond brace placement);
   e. env read in `ggml_backend_sched_new` (coexists with leloch's
      `session_create` block above it);
   f. free block in `ggml_backend_sched_free`; sync of `prefetch_backend` in
      `ggml_backend_sched_synchronize`.
4. Optional guard to add during 3c: skip prefetch when the tensor is managed by an
   active moe-cache session (see divergence note 3).
5. Docs: add both env vars to a short section (fork README has a table we can
   mirror, ASCII-only per AGENTS.md; leloch docs live under plans/ + docs).

### Verification gates

- Clean CPU build + `ctest` (expect 60/60 like the v4 port report).
- CUDA build (compile-only is possible here - the v4 report notes no GPU on this
  machine; runtime benchmarking must happen on a GPU box).
- Correctness: with both env vars unset, output must be bit-identical to branch
  baseline (features are pure opt-in; assert by diffing `-p 512 -n 32` greedy
  outputs).
- Correctness with flags ON: greedy decode token-identical to baseline at
  `-ub 512` and `-ub 2048` (prefetch only reorders uploads; 5f83fbbe7's whole point
  was that a fallback eval must not see a stale slot). Add GGML_SCHED_DEBUG smoke
  runs and an OOM-arm test (`GGML_SCHED_PREFETCH_EXPERTS=8` with a tight
  `--n-gpu-layers` to force slot-allocation failure -> expect graceful disable,
  no crash).
- Perf: `llama-bench -m <MoE gguf> -ngl 99 -ncmoe <N> -p 2048 -n 0 -r 5 -b 2048
  -ub 2048`, arms: baseline / REGISTER_HOST / PREFETCH / both; plus a decode arm
  (`-p 0 -n 128`) to confirm decode is untouched; plus a moe-cache-ON arm to
  characterize the interaction.
- Watch item: moe-cache fill path + pinned pages interaction (fills get faster;
  pinning large ranges could contend with the cache budget on `cudaHostRegister`
  lock memory - log line prints pinned MiB, check it against expectations).

### Effort estimate

- Feature 1: cherry-pick, <1h including build check.
- Feature 2: ~200-line manual transplant into a rewritten function, 0.5-1 day with
  the conflict archaeology and the moe-cache gating decision.
- Benchmark/validation sweep: the long pole, and it needs a GPU machine.

## Open questions for the user

1. Which model/GPU will we validate on? Fork numbers are RTX 3060 + Qwen3.6-35B-A3B;
   the v4 branch work targeted the moe-cache workload set in
   plans/leloch-v4-port-report.md.
2. Should prefetch stay strictly env-gated (recommended, matches fork and this
   repo's opt-in style) or get a CLI flag?
3. Gate prefetch off when a moe-cache session is active (safe first cut) or invest
   in making the two compose from day one?
