# Prefill env vars: GGML_CUDA_REGISTER_HOST + GGML_SCHED_PREFETCH_EXPERTS

Both features speed up prefill of MoE models whose routed experts stay in system
RAM (`--n-cpu-moe` / `--cpu-moe`) by making the host-to-device weight streaming
path faster. Both are opt-in: unset (default) = unchanged behavior, bit-identical
output. Relevant on CUDA/HIP/MUSA backends; decode is unaffected (the prefetch
heuristic self-excludes small batches).

| Env var                       | Values                                       | Default | What it does |
| ---                           | ---                                          | ---     | --- |
| `GGML_CUDA_REGISTER_HOST`    | `1`                                          | unset   | Page-locks (`cudaHostRegister`) the retained mmap byte-range that backs CPU-resident weights after model load, so all H2D copies from it use DMA from pinned pages instead of the driver's pageable bounce path. |
| `GGML_SCHED_PREFETCH_EXPERTS` | `1` -> 3 slots, `N` > 1 -> N slots (capped at 8) | unset   | Uploads the full host-resident `MUL_MAT_ID` expert tensor on a second backend instance (own stream) one tensor ahead of compute, overlapping upload N+1 with compute N, instead of syncing to read routing ids and issuing partial expert copies. |

## GGML_CUDA_REGISTER_HOST

Wiring (src/llama-mmap.cpp, src/llama-model-loader.cpp):

- After `load_all_data()` unmaps the unused head/tail fragments, each retained
  mapping range is expanded outward to page boundaries and registered via the
  first backend registry exposing `ggml_backend_register_host_buffer`
  (CUDA/HIP/MUSA). Registration uses
  `cudaHostRegisterPortable | cudaHostRegisterReadOnly`.
- Success logs `pinned %.2f MiB of mapped model memory ...` at info level.
- `~llama_mmap()` unregisters before the pages are unmapped.

Requirements and failure mode:

- Page-locking multi-GB ranges requires sufficient `RLIMIT_MEMLOCK` or
  `CAP_IPC_LOCK` (e.g. `ulimit -l unlimited`, or run the process with the
  capability). On failure `register_host` returns 0 and everything keeps working
  over the pageable path (graceful no-op; the driver error shows up at
  GGML_LOG_DEBUG in ggml-cuda.cu).

Usage:

```sh
ulimit -l unlimited
GGML_CUDA_REGISTER_HOST=1 llama-cli -m model.gguf -ngl 99 -ncmoe 26 -p 2048 -n 0
```

## GGML_SCHED_PREFETCH_EXPERTS

Implementation in ggml/src/ggml-backend.cpp (`ggml_backend_sched_compute_splits`):

- Values: `1` enables the default of 3 slots (the gate/up/down tensors of one MoE
  layer); `N` > 1 sets N slots directly, capped at `GGML_SCHED_MAX_PREFETCH_SLOTS`
  (8). Each slot costs one max-sized expert tensor of device memory.
- Requires `op_offload` (partial offload, i.e. `-ncmoe` / `-cpu-moe` in use).
- Eligibility per split: host WEIGHTS-buffer input, first node is `MUL_MAT_ID`
  with `src[0] == input_cpy`, and dense batch (`ids->ne[0] * ids->ne[1] >=
  2 * n_expert`) - so decode and sparse batches never take this path.
- Skipped while a moe-cache session is active (`!sched->moe_cache_session`
  guard). The ids-readback partial-copy fallback stays fully intact: when
  prefetch is off, ineligible, or disabled mid-run, behavior is identical to
  mainline.
- Slots are sized once for the largest offloaded expert tensor. On allocation
  failure the feature degrades to fewer slots if at least 2 fit, else disables
  itself (syncs both backends, frees slots) and falls back - no crash.
- The staging tensor is pointed at the slot only for the duration of the split
  and restored right after launch, so graph reuse or a later fallback eval can
  never observe a dangling slot pointer.

Usage:

```sh
# default 3 slots
GGML_SCHED_PREFETCH_EXPERTS=1 llama-bench -m model.gguf -ngl 99 -ncmoe 26 -p 2048 -n 0 -ub 2048

# both features together
GGML_CUDA_REGISTER_HOST=1 GGML_SCHED_PREFETCH_EXPERTS=1 llama-server -m model.gguf -ngl 99 -ncmoe 26
```

Note: the fork also carries `GGML_SCHED_ASYNC_CPU`; that is a separate feature and
was deliberately NOT ported here.
