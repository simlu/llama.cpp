# Port report: prefill optimizations on leloch-v4-prefill

Branch: leloch-v4-prefill (from leloch-v4 f1eaeab21)
Date: 2026-10-02
Machine: 16 cores, CUDA 13.4 at /usr/local/cuda, NO GPU (CUDA compile-only verification)

## What was ported

Two env-gated prefill optimizations from thecodacus/llama.cpp (local ref
`thecodacus/perf`), plan in plans/prefill.md:

| New SHA   | Source                                       | What |
| ---       | ---                                          | --- |
| b98830ed8 | cherry-pick -x 20f5994bf                     | llama : pin mmap-backed CPU weights for faster H2D uploads (`GGML_CUDA_REGISTER_HOST` caller-side wiring; src/llama-mmap.{h,cpp}, src/llama-model-loader.cpp, +59/-1) |
| 39992f063 | manual port of 1163cb349 + 5f83fbbe7 final shape | ggml : overlap offloaded expert weight uploads with compute (`GGML_SCHED_PREFETCH_EXPERTS`; ggml/src/ggml-backend.cpp only, +173/-0) |

The CUDA-side gate functions (`ggml_backend_cuda_register_host_buffer` in
ggml-cuda.cu) already existed on leloch-v4 verbatim from upstream d0a71233f; the
cherry-pick only adds the callers. No ggml-cuda.cu changes.

## Deviations from the fork shape

- Prefetch eligibility adds `!sched->moe_cache_session`: prefetch is skipped while
  a leloch moe-cache session is active. The fork has no such guard (its MoE cache
  diverged). This is the "safe first cut" option from the plan; relax behind a
  flag only if benchmarks show the two compose.
- The fork's `GGML_SCHED_ASYNC_CPU` (cpu_async) weaves through the same function
  and was NOT ported.
- Fork commits like 08fa762f7 (expert-pack split predicate) and the fork's own
  MoE-cache commits belong to other fork work; not touched.
- The cherry-pick conflict in llama-mmap.cpp was ctor-signature drift only (the
  leloch `lazy_ranges` param); resolved by keeping the leloch signature.

## Verification results

From the parent tickets (both on HEAD 39992f063):

- CPU: clean build exit 0; `ctest` 60/60 green with both env vars unset
  (no-op sanity satisfied). Both env gates confirmed present in source.
  `git diff leloch-v4..HEAD` ASCII-clean. No fix commits needed.
- CUDA: compile-only build green (`cmake --build build-cuda -j`, GGML_CUDA=ON,
  CUDA 13.4, exit 0). No GPU on this machine - runtime NOT verified.

## Remaining GPU benchmark TODO (from plans/prefill.md "Verification gates")

All arms need a GPU box:

- [ ] llama-bench pp2048 (`-ngl 99 -ncmoe N -p 2048 -n 0 -r 5 -b 2048 -ub 2048`),
      arms: baseline / `GGML_CUDA_REGISTER_HOST=1` / `GGML_SCHED_PREFETCH_EXPERTS=1`
      / both. Fork reference: RTX 3060, Qwen3.6-35B-A3B, `-ncmoe 26`:
      1143 -> 1385 -> 1880 t/s (+64%); unverified here.
- [ ] Decode arm (`-p 0 -n 128`) to confirm decode is untouched.
- [ ] moe-cache-ON arm to characterize the interaction with the moe_cache_session
      guard (does prefetch ever fire alongside the cache; is the guard too strict).
- [ ] Correctness: greedy outputs (`-p 512 -n 32`) bit-identical to baseline with
      both vars unset; token-identical with flags ON at `-ub 512` and `-ub 2048`.
- [ ] OOM arm: `GGML_SCHED_PREFETCH_EXPERTS=8` with a tight budget to force slot
      allocation failure -> expect graceful disable, no crash.
- [ ] Watch: pinned MiB log line vs expectations; interaction of large
      `cudaHostRegister` ranges with the moe-cache fill path / lock-memory budget.

Nothing was pushed and no PR was opened.
