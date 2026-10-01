# Port report: GGML_CUDA_REGISTER_HOST + GGML_SCHED_PREFETCH_EXPERTS

Branch: leloch-v3-prefill (base = leloch-v3 moe-cache tip fd349cf6a + plan doc 405693f98)
Source fork: https://github.com/thecodacus/llama.cpp, branch `perf`
Date: 2026-10-01
Machine: 16 cores, nvcc 12.4.131, NO GPU (CUDA compile-only verification)

## Summary

Two self-contained, opt-in fork features targeting the `--n-cpu-moe`
(-ncmoe) path were cherry-picked onto leloch-v3-prefill with `-x`
provenance: Feature A (GGML_CUDA_REGISTER_HOST, 1 commit) and Feature B
(GGML_SCHED_PREFETCH_EXPERTS, 2 commits). Both are env-var gated, so the
default behavior is unchanged. CPU build passes (100% targets), CUDA
compile-only build passes (100% targets), and the branch diff is clean on
`git diff --check`. CUDA runtime is NOT verified: this machine has no GPU.
Nothing was pushed and no PR was opened.

## 1. What was ported

All three commits were cherry-picked onto leloch-v3-prefill with the
`(cherry picked from commit ...)` provenance line intact. Branch SHA ->
source fork SHA:

| Branch SHA | Source SHA | Subject |
| --- | --- | --- |
| 26bf8d712 | 20f5994bf | llama : pin mmap-backed CPU weights for faster H2D uploads |
| 4759cbd70 | 1163cb349 | ggml : overlap offloaded expert weight uploads with compute |
| 7da5b3012 | 5f83fbbe7 | ggml : size prefetch slots per layer and fix fallback use-after-free |

Feature A (GGML_CUDA_REGISTER_HOST) is commit 26bf8d712 (source 20f5994bf),
a call-site-only wiring change. The backend function
`ggml_backend_cuda_register_host_buffer()` and its env gate already existed
in this branch from upstream (commit 47373271f HIP enable + c3d6af7cd CUDART
fix) at ggml/src/ggml-cuda/ggml-cuda.cu:4939 and :4962, so only the calling
site was ported (llama-mmap.h, llama-mmap.cpp, llama-model-loader.cpp).
No change was made to ggml-cuda.cu.

Feature B (GGML_SCHED_PREFETCH_EXPERTS) is commits 4759cbd70 (source
1163cb349, v1) and 7da5b3012 (source 5f83fbbe7, v2). Both are in
ggml/src/ggml-backend.cpp. v2 fixes a use-after-free (staging repoints were
restored after kernel launch) and bumps the default slot count from 2 to 3;
both commits are required. No other fork commits were brought in (the
fork's `cpu_async` / GGML_SCHED_ASYNC_CPU and its MoE expert-cache commits
are not part of this port).

## 2. Env gating

Both features are opt-in via environment variables; with them unset, the
default runtime behavior is unchanged.

GGML_CUDA_REGISTER_HOST:
- Gate confirmed at ggml/src/ggml-cuda/ggml-cuda.cu:4940 and :4963
  (`getenv("GGML_CUDA_REGISTER_HOST") == nullptr` -> return).
- Semantics: when experts are kept in system RAM (--n-cpu-moe), the model
  weights live in mmap'd pageable file pages. cudaHostRegister() page-locks
  those pages so host->device copies go straight over DMA instead of the
  driver's pageable bounce buffer. Setting GGML_CUDA_REGISTER_HOST=1 pins
  the mmap pages backing weights retained in host memory; the pages are
  unregistered in the llama_mmap destructor. Unset by default.

GGML_SCHED_PREFETCH_EXPERTS:
- Gate confirmed at ggml/src/ggml-backend.cpp:2179-2180
  (`getenv("GGML_SCHED_PREFETCH_EXPERTS")`).
- Semantics: above a batch threshold it skips the per-layer routing-id
  readback and uploads the full offloaded expert tensor through a second
  backend instance on the same device (its own stream), alternating among
  event-ordered staging slots so upload for tensor N+1 overlaps compute of
  tensor N. Slot count: default 3, configurable to a max of 8
  (GGML_SCHED_MAX_PREFETCH_SLOTS = 8 at ggml-backend.cpp:840). At
  ggml-backend.cpp:2183, `prefetch_n_slots <= 1 ? 3 :
  std::min(prefetch_n_slots, GGML_SCHED_MAX_PREFETCH_SLOTS)`. Unset by
  default.

Both env vars are implementation controls, not new CLI flags; `--help` is
unchanged.

## 3. Build and verification results

All results below were produced on this machine (no GPU).

CPU build (GGML_CUDA=OFF), build-cpu:
- Fresh configure/build after Feature A (Step 2) and after Feature B
  (Step 4): 100% targets, exit 0.
- Incremental rebuild this run: 100% targets, exit 0 (clean no-op).

CUDA compile-only (GGML_CUDA=ON), build-cuda (Step 5):
- cmake -B build-cuda -DGGML_CUDA=ON -DLLAMA_BUILD_TESTS=ON
  -DCMAKE_BUILD_TYPE=Release; cmake --build build-cuda -j16: 100% targets,
  exit 0. libggml-cuda.so.0.25.3 built.
- The 3 ported host files recompiled (ggml-backend.cpp.o, llama-mmap.cpp.o,
  llama-model-loader.cpp.o) and the CUDA sources ggml-cuda.cu.o +
  moe-cache.cu.o recompiled.
- Incremental rebuild this run: 100% targets, exit 0 (clean no-op).

Static checks:
- git diff --check origin/master..HEAD: exit 0 (clean).
- git diff --check HEAD~3..HEAD (the 3 prefill commits only): exit 0 (clean).
- ASCII: the prefill-only diff (HEAD~3..HEAD) contains 0 non-ASCII bytes.
  The full branch diff vs origin/master contains exactly 2 non-ASCII bytes
  (2x "+/-" in a llama-bench README benchmark table); those are inherited
  verbatim from the moe-cache port's README rewrite and are NOT introduced
  by the prefill commits.

No repo changes were made beyond the 3 commits and this report (working
tree clean).

## 4. Known limitations

- No GPU on this machine (/dev/nvidia* absent, no nvidia-smi), so the DMA
  and stream runtime behavior, correctness, and the measured performance of
  both features are UNVERIFIED. The moe-cache report flagged the same
  limitation. This machine verified compilation and static checks only.
- Prefetch requires `caps.events` (peer-copy-enabled CUDA). On builds with
  GGML_CUDA_NO_PEER_COPY, caps.events is false and prefetch silently
  disables itself. This is by design.
- VRAM cost: prefetch allocates prefetch_n_slots (default 3, max 8) staging
  buffers, each sized to the largest offloaded expert tensor
  (ggml_backend_sched_prefetch_max_size). On a small GPU this can OOM; the
  code degrades by shrinking the slot count (ggml-backend.cpp:1813,
  `sched->prefetch_n_slots = i`) and falls back to the expert-copy path.
- The branch's moe-cache keeps hot experts VRAM-resident while prefetch
  streams the cold CPU remainder; this combination is the least-tested
  path and must be validated on real hardware before any performance claim.

## 5. Rollback

Nothing is pushed and no PR exists. Each commit carries its
`(cherry picked from commit ...)` provenance line, so the port can be
reverted per-feature via `git revert <sha>` or the whole branch reset to
fd349cf6a. The 3 branch SHAs to revert:

- 26bf8d712 (Feature A, GGML_CUDA_REGISTER_HOST; source 20f5994bf)
- 4759cbd70 (Feature B v1, GGML_SCHED_PREFETCH_EXPERTS; source 1163cb349)
- 7da5b3012 (Feature B v2, slot-count fix + use-after-free; source
  5f83fbbe7)

The fork reference commits and the /tmp/thecodacus-llama clone remain
available for re-inspection.

## 6. Hardware handoff checklist

All items below require a real GPU and must be recorded with GPU model(s),
PCIe topology, CPU model, memory channels and speed, model quantization,
exact placement, build revision, and environment overrides. Fork reference
numbers (RTX 3060, Qwen3.6-35B-A3B, pp2048, -ncmoe 26, ub 2048):
mainline 1143 t/s -> register 1385 t/s -> both on 1880 t/s.

1. llama-bench, 3 arms at pp2048 with `-ncmoe <N> -ngl 99`:
   - baseline: both env vars off
   - arm 2: GGML_CUDA_REGISTER_HOST=1 only
   - arm 3: both GGML_CUDA_REGISTER_HOST=1 and GGML_SCHED_PREFETCH_EXPERTS=1
   Keep placement/context/batch/threads/warmup identical across arms.
2. Verify token-identical output vs baseline: a prefill pass and a decode
   pass (both features must not change the result).
3. Test the prefetch degradation path on a small VRAM budget: force an OOM,
   confirm the slot count shrinks (ggml-backend.cpp:1813) and that there is
   no crash and no use-after-free (the v2 fix repoints staging tensors
   right after kernel launch).
4. Confirm `--help` shows no new CLI flags (both features are env-var only).

## Rollback summary

The branch state is fully described by this report. Each commit retains its
provenance line, so the port can be reproduced, inspected, or reverted at
any time.
