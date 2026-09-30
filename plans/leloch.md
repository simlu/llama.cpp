# Plan: cherry-pick leloch/moe-cache-v2-pr onto simlu/llama.cpp master

Goal: port the 29 commits of https://github.com/leloch/llama.cpp/tree/moe-cache-v2-pr
onto current origin/master (simlu/llama.cpp) in a new branch `leloch-v3`.
No code changes have been made yet; this document is the planning deliverable.

---

## 1. Source branch facts

- Branch: `leloch/moe-cache-v2-pr`, tip `e3096b046` ("cuda: pack MoE cache dispatch inputs")
- Author: leloch (13730836+leloch). Tip date 2026-08-06.
- 29 commits, strictly linear (0 merge commits), all by leloch.
- Branch base (merge-base with origin/master): `15586e2d7` ("mtmd: add chunk save/load
  function (#26645)"), dated 2026-08-06. So the whole series was built on a ~2-month-old
  upstream snapshot.
- Full branch delta vs its base: 28 files, +9289 / -312.
- Branch history is clean/ASCII-only (no non-ASCII chars; matches AGENTS.md style rules).

Upstream leloch fork also has: `master` (stale, ~#24395 era), `moe-cache` / `moe-cache-pr`
(older designs), `v3-expert-cache` (the "ec3" scheduler/fusion/persistence design that
commit f2d7f9303's message says it *replaces*; based on an even older upstream commit).
None of them is a newer version of this branch. `moe-cache-v2-pr` is the newest design
and the correct source.

## 2. What the feature is and why (the "MoE expert cache")

Problem: MoE models (DeepSeek V4, GLM-5.2, gpt-oss, Qwen3-MoE, ...) have routed expert
weights far larger than VRAM (GLM-5.2 UD-IQ2_M: ~209 GiB of experts). They run with
experts in host memory and compute the per-token expert matvecs on the CPU via
`MUL_MAT_ID` (vec_dot per expert per row). Decode is memory-bandwidth-bound on that path,
which caps throughput (cache-off controls measured 14-23 t/s on 4x RTX 3090).

Insight: the routing set is small and sticky (a few dozen hot experts carry most hits).
So: keep the hottest experts resident on CUDA and run their matvecs on GPU *in parallel
with* the CPU computing the miss rows of the same node ("per-node hybrid execution").
Measured gains (from the branch docs, 4x RTX 3090):
- DeepSeek V4 Q8_K_XL canonical weights: +18.3% (1 GPU) up to +48.0% (4 GPUs).
- GLM-5.2 UD-IQ2_M with MTP: 14.46 t/s -> 28.25 t/s (production profile).
- Quality: perplexity statistically equivalent (2.7996 off vs 2.7987 on; 2.2090 vs 2.1960),
  long-context retrieval correct. Output is NOT bit-identical (CPU vs CUDA rounding can
  change near-tie token decisions) - by design.

The cache is opportunistic: unsupported nodes, no capacity, contention, or any failure
falls back to the stock CPU path (correctness is preserved; performance only).

### Architecture / integration map

New files (no git conflicts possible, but API-drift compile risk):
- `ggml/src/ggml-backend-moe-cache.h` - public C API table `ggml_moe_cache`:
  query_config/query_device/query_shape, session_create/destroy/enter/leave,
  per-node begin/plan/dispatch/collect/end, and invalidate(host base,size).
- `ggml/src/ggml-cuda/moe-cache.cu` (~3145 lines) - session/pool machinery: device caps,
  shape inventory, pool allocation, slot pinning + generation tracking, per-device
  dispatch locks, bounded demand-fill worker threads, LRU eviction w/ throttle,
  scratch budgeting, allocator-pressure trim, teardown, stats. Env-tunable
  (GGML_CUDA_MOE_CACHE_* family: MODE, BUDGET_MB, RESERVE_MB, MIN_EXPERT_KB, MAX_BATCH,
  INSERTS, ADMIT_AFTER, THROTTLE, QUEUE, QUEUE_MB, NDEV, MIN_CC, SERIAL_FILL,
  DEDICATED_MMV, OVERLAP_CPU_ROWS, FAIL, STATS).
- `ggml/src/ggml-cuda/moe-cache.cuh` - register/trim entry points.
- `tests/test-moe-cache.cpp` (3154 lines) - lifecycle/concurrency/invalidation/fallback/
  routing/admission tests; needs a CUDA device, skips (exit 0) when absent.
- `tests/test-moe-cache-fit.cpp` (170 lines) - pure-CPU tests of the fit planner.
- `docs/backend/CUDA-MOE-CACHE.md` (261 lines) - full documentation.

Modified integration points:
- `ggml/src/ggml-backend.cpp` - defines the global `ggml_moe_cache` table +
  `ggml_moe_cache_unregister()`; adds invalidation hooks on every public write path for
  host WEIGHTS buffers (buffer_free/clear/set_usage/reset, copy_tensor, set(_async),
  set_2d(_async), memset, tensor_copy(_async)); adds `moe_cache_session` to
  `ggml_backend_sched`, creates it in `ggml_backend_sched_new`, scopes
  enter/leave around `ggml_backend_sched_compute_splits`, exposes
  `ggml_backend_sched_set_moe_cache()` and destroys the session in sched_free.
- `ggml/src/ggml-cpu/ggml-cpu.c` - inside `ggml_compute_forward_mul_mat_id`: thread 0
  calls begin/plan for eligible nodes (weight tensor name contains `_exps`, host
  WEIGHTS buffer, src0 op NONE, F32 activations, <= 64 routed rows), records hit rows
  (slot index >= 0), removes them from the CPU matrix-row work, dispatches to GPU,
  then collects results after CPU workers; on dispatch/collect failure the rows are
  restored to the CPU mapping (correct fallback). The fusion commit (098f547aa)
  refactors this into `..._impl()` with row_mask + allow_moe_cache and adds graph-level
  gate/up/SwiGLU subgraph detection so fully-resident gate+up rows run a fused
  SwiGLU kernel.
- `ggml/src/ggml-cuda/mmvq.cu` + `.cuh` - threads an `act_ids_ptr` through the
  mul_mat_vec_q kernel launch path (dispatch of specific expert rows), adds
  `allow_small_k` control, bounds tail-row reads (512b2fb6d), and adds the pool
  matvec kernels `ggml_cuda_moe_cache_mmv` / `_fused` (+ templates).
- `ggml/src/ggml-cuda/ggml-cuda.cu` - cache trim hook on allocation failure
  (cudaMalloc / cuMemCreate retry after `ggml_moe_cache_trim(device)`), and
  `ggml_moe_cache_register(&reg)` at backend registration (CUDA only; HIP/MUSA compile
  inert stubs).
- `ggml/src/ggml-backend-reg.cpp` - `ggml_moe_cache_unregister(reg)` on backend unload.
- `include/llama.h` - `llama_moe_cache_mode` enum + `moe_cache_mode` /
  `moe_cache_budget_mib` fields in `llama_context_params`.
- `src/llama-cparams.h`, `src/llama-context.cpp` - cparams fields; applied via
  `ggml_backend_sched_set_moe_cache` after sched creation/reserve (3 call sites).
- `src/llama-ext.h`, `src/llama-model.cpp` - `llama_model_get_moe_tensor_info()`:
  inventories `*_exps.weight` / `*_chexps.weight` tensors (type, expert size, shapes)
  for cache-aware fit; uses `model->tensors_by_name` (present in master).
- `common/fit.h` / `common/fit.cpp` - cache-aware fit: `common_moe_cache_plan_fit()`
  (device aggregation by physical device, per-device cache budget = free - reserve,
  shape inventory, 64-slot floors, scratch accounting), and a new
  `common_moe_cache_params *` parameter added to `common_fit_params()` /
  `common_get_device_memory_data[_with_parent]` so auto-fit can choose
  canonical-CPU-expert placement and reserve VRAM for cache pools.
- `common/common.h` / `common/arg.cpp` / `common/common.cpp` - `common_moe_cache_params`
  in `common_params`; `--moe-cache MODE` option (auto/on/N/off/0) with env
  `LLAMA_ARG_MOE_CACHE`; mapping into cparams + informational logging; explicit cache
  mode disables weight repacking (`no_extra_bufts`).
- `tools/server/server-context.cpp` - DSpark/DFlash shared-draft device auto-placement
  (pick GPU with most free memory, exclude main GPU when possible), MTP/DRaft fit
  reservation accounting (reserve per-device max for the sidecar before fitting),
  DSpark device selection fix, hardened placement logic.
- `tools/llama-bench/llama-bench.cpp` + README - `--moe-cache` / `--repack on|off`
  bench arms; README documents the 3-arm measurement procedure.
- `tools/fit-params/fit-params.cpp` - passes moe_cache params into fit and echoes the
  effective `--moe-cache`/`--no-repack` in the printed command.
- `tests/CMakeLists.txt`, `tests/test-arg-parser.cpp` - build registration + parser tests.

### Commit list (chronological; all 29)

1. f2d7f9303 cuda: rework MoE expert cache execution  (+6814/-279) - THE base commit:
   entire feature in its first "rework" iteration (replaces the older ec3/v3 design:
   no hot-set persistence, no fused GLU/GPU-output handoff, no auto-fit GGUF scan).
2. bb1ea6933 cuda: fix MoE cache capacity accounting   (moe-cache.cu + tests)
3. 986036549 server: fix auto-selected DSpark device    (server-context.cpp)
4. 6d20f3770 docs: update MoE cache validation          (docs)
5. 4120695e0 common: preserve implicit MoE cache provider settings
   (common.cpp, docs, test-arg-parser.cpp) - mode_explicit gating: only explicit CLI/env
   selection overrides; fit-selected placement no longer leaks into cparams silently.
6. 241372ae5 server: harden shared draft device placement (server-context.cpp +307)
7. f04585925 cuda: fix MoE cache scratch budget replacement (moe-cache.cu)
8. b7024cc10 llama: skip dormant MoE cache sessions      (llama-context.cpp +49)
9. 512b2fb6d cuda: bound MoE MMV tail-row reads         (mmvq.cu, 3 lines)
10. f0bd8b1b5 cuda: tune MoE cache CPU overlap automatically (moe-cache.cu, header, tests)
11. 24ae6636b cuda: adapt MoE cache admission to device capability
    (fit.cpp/h, header, moe-cache.cu, llama-context.cpp, tests)
12. 43c3be2af cuda: align MoE cache pool allocation with fit
    (fit.cpp, header, moe-cache.cu, llama-context.cpp, tests)
13. d5ffc999f cuda: prefer generic MMV for compatible MoE cache nodes (moe-cache.cu)
14. 135e204d2 cuda: parallelize MoE cache fills on capable multi-GPU systems
    (moe-cache.cu, tests)
15. adbe59d1d server: account for MTP placement in fit reservations
    (fit.cpp/h + server-context.cpp)
16. a38ba195a docs: update MoE cache validation (docs)
17. 098f547aa cuda: fuse cached MoE SwiGLU rows (+1736) - the biggest follow-up:
    SwiGLU subgraph fusion for fully-resident gate+up rows (ggml-cpu.c +359,
    mmvq.cu +146, moe-cache.cu +750, tests +655).
18. 32fbcd681 cuda: aggregate small MoE tensors into cache pools (fit.cpp, tests)
19. c0d7d916b cuda: bound automatic MoE cache admission (moe-cache.cu, tests)
20. a68d9dad6 cuda: reject undersized automatic MoE cache slabs
    (fit.cpp/h, header, moe-cache.cu, llama-context.cpp, tests)
21. 9aeef2cfd cuda: enable multi-token MoE cache in forced modes (moe-cache.cu, tests)
22. 18b913e29 common: surface MoE cache activation (common.cpp, moe-cache.cu, docs, tests)
23. 7d4af9e6e docs: fix MoE cache benchmark verbosity (docs)
24. b07c53689 cuda: accelerate complete MoE cache pools (moe-cache.cu, tests)
25. 107b9d596 cuda: report oversized MoE cache nodes
    (header, ggml-cpu.c, moe-cache.cu, tests, docs)
26. 78104ec46 cuda: clarify MoE cache session diagnostics (moe-cache.cu, docs)
27. fd612c058 docs: record MoE cache workload convergence (docs)
28. f31e9b7c7 cuda: report MoE cache pair residency (moe-cache.cu, docs)
29. e3096b046 cuda: pack MoE cache dispatch inputs (moe-cache.cu)

## 3. Current repo state (origin/master)

- Tip: 19e28a277 (2026-09-30), ~2 months / ~300+ commits ahead of the branch base.
- Only working-tree change: `.gitignore` adds `/.hermes-memory/` (pre-existing, local
  only). Leave it untouched; do not include it in any commit.
- No CUDA toolchain on this machine (no nvcc, no GPU; 16 cores, 62 GB RAM, 17 GB free disk).

### Divergence: master-side changes (15586e2d7 -> origin/master) per touched file

(columns: branch delta vs master delta in changed lines; both from git numstat)

| file | branch | master | risk | notes |
|---|---|---|---|---|
| tools/server/server-context.cpp | +306/-29 | 1779 | HIGH | 24 master commits: batch_ext migration, ctx-per-slot, mmproj device fit, sleep refactor, yield_to_queue thread model, metrics. Branch touches draft-device placement + fit reservations - same regions. |
| src/llama-context.cpp | +100 | 717 | HIGH | 30 commits: llama_batch_ext, W4A4/prec_policy, CUDA graph MTP, multi-output sampling. Branch's 3 set_moe_cache sites are in constructor + sched_reserve (2x) + default params; master's sched_reserve still has the same 2 sched.reset call sites. |
| src/llama-model.cpp | +36 | 579 | LOW | Branch only appends llama_model_get_moe_tensor_info; tensors_by_name exists. |
| ggml/src/ggml-cuda/ggml-cuda.cu | +21 | 558 | MED | Trim hooks in ggml_cuda_device_malloc region + cuMemCreate + reg init; master's alloc code structurally similar (same look_ahead_size pattern) but evolved (35 commits). |
| common/common.cpp | +29 | 504 | LOW | cparams mapping hunk lands right after type_k/type_v assignment - byte-identical region in master. |
| ggml/src/ggml-cuda/mmvq.cu | +299 | 360 | HIGH | Master reworked launch machinery: mmvq_parameter_table_id, mmvq_should_prefetch, halve_iters (GB10), small_k promotion, mul_mat_vec_q_moe<bool has_fusion>; branch threads act_ids_ptr + allow_small_k through the same functions + adds pool MMV kernels. Manual port needed. |
| common/arg.cpp | +38 | 334 | LOW-MED | --moe-cache option is additive; check overlap with spec/vision/n-cpu-ffn args added in master. |
| ggml/src/ggml-backend.cpp | +130/-2 | 200 | MED | 13 commits; write-path signatures must match (buffer_set_usage, cpy_tensor etc.); verify master didn't change those function bodies' shapes. |
| include/llama.h | +10 | 172 | LOW-MED | Additive enum + 2 cparams fields after type_v; check what master added to llama_context_params. |
| common/fit.cpp | +463/-43 | 163 | MED-HIGH | Master: n_streams ctx logic, LOG_JSON, auto-fit max-ctx revert (#29437). Branch adds ~230 lines at file top + modifies common_params_fit_impl (overlapping regions 190-560, 805-1192). |
| tools/llama-bench/llama-bench.cpp | +307/-35 | 157 | MED | Branch converts repack to string on/off + moe_cache options; master: hf_file OOB fix, --version, defaults. Same arg-parsing regions. |
| common/common.h | +14 | 157 | LOW | Field added at end of common_params; master added fields too (batch_ext etc.) - minor context shift. |
| ggml/src/ggml-cpu/ggml-cpu.c | +445/-3 | 51 | MED | Master: tiled mul_mat k-quants, FWHT F16, SWIGLU_CLAMP, AVX2 IQ - mostly other areas; mul_mat_id structure still matches (one_chunk helper at ~same place). |
| tests/CMakeLists.txt | +7 | 47 | LOW | Additive registration; master's test registration style unchanged. |
| ggml/src/ggml-backend-reg.cpp | +3 | 18 | LOW | 2-line hunk in unload path. |
| common/fit.h | +62 | 11 | LOW | But common_fit_params() signature change -> 3 call sites in master: common/common.cpp:1225, tools/fit-params/fit-params.cpp:34, tools/llama-bench/llama-bench.cpp:2346. |
| src/llama-ext.h | +13 | 2 | LOW | Additive. |
| src/llama-cparams.h | +3 | 2 | LOW | Additive. |
| tests/test-arg-parser.cpp | +126 | 2 commits | LOW | Additive tests. |
| tools/fit-params/fit-params.cpp | +13/-1 | 1 | LOW | Call-site update. |
| tools/llama-bench/README.md | 192 | 19 | LOW | Docs; merge both sides. |
| new files (moe-cache.cu/cuh, ggml-backend-moe-cache.h, test-moe-cache*.cpp, CUDA-MOE-CACHE.md) | - | - | n/a | No git conflict, but compile-drift risk against master's CUDA internals (ggml_backend_cuda_context, ggml_cuda_info, ggml_cuda_set_device, mmvq.cuh decls - all still exist in master; verified). |

Not touched by master but touched by branch: none missing; all branch target files still
exist in master with the same paths.

### Specific API-drift checks already done (all pass)

- `ggml_cuda_info()` / `ggml_cuda_device_info` still exist (common.cuh:1168/1193).
- `ggml_backend_cuda_context` still exists (common.cuh:1446).
- `ggml_cuda_mul_mat_vec_q` still exists with the same 5-arg signature.
- `ggml_cuda_device_malloc` + `look_ahead_size` + `cuMemCreate` regions still present.
- `ggml_backend_cuda_reg` still present.
- `tensors_by_name` map present in llama-model.cpp.
- b5cf8ce02 (master, "require input tensors to be GGML_OP_NONE") makes the branch's
  `src0->op == GGML_OP_NONE` eligibility check consistent with master semantics - good.

## 4. Cherry-pick strategy

Branch: create `leloch-v3` from `origin/master` (currently at 19e28a277).
Order: strictly chronological (f2d7f9303 first), `git cherry-pick -x` for provenance.
Never commit the .gitignore change; never push; no PR (AGENTS.md rules).

Phase A - base commit f2d7f9303 (the big one, +6814):
- Apply; expected conflicts in: ggml-cpu.c, mmvq.cu, ggml-cuda.cu, ggml-backend.cpp,
  llama-context.cpp, llama-model.cpp, common.cpp, common.h, arg.cpp, fit.cpp, fit.h,
  llama.h, llama-ext.h, llama-cparams.h, llama-bench.cpp, server-context.cpp,
  fit-params.cpp, tests/CMakeLists.txt, test-arg-parser.cpp.
- Resolution policy per file:
  * keep branch semantics, but adapt anchors to master's structure (master is the base).
  * mmvq.cu: port act_ids_ptr/allow_small_k threading onto master's new launch machinery
    (halve_iters, parameter tables). This is the hardest piece; the branch's final
    mmvq.cu (tip) is the reference for intended behavior - diff tip-mmvq.cu against
    master-mmvq.cu and re-express the deltas instead of applying hunks mechanically.
  * server-context.cpp: port only what the branch adds (shared-draft device placement,
    fit reservation for sidecars). Master's fit flow (mmproj fit additions around
    line 1045-1075, MTP context fitting) must be preserved.
  * fit.cpp: keep master's n_streams/LOG_JSON changes; insert branch's planner
    (common_moe_cache_plan_fit + integration into common_params_fit_impl +
    common_fit_params signature) with new call-site updates in common.cpp,
    fit-params.cpp, llama-bench.cpp.
- After Phase A: full CPU build + CPU test suite must pass before continuing.

Phase B - the 28 follow-ups, in order:
- New-file commits (moe-cache.cu, tests, docs, header) apply cleanly since those files
  exist in our tree in branch-equivalent state after Phase A.
- Old-file follow-ups and where to expect conflicts:
  * llama-context.cpp: b7024cc10 (+49), 24ae6636b, 43c3be2af, a68d9dad6 (small).
  * fit.cpp/fit.h: 32fbcd681, a68d9dad6, 43c3be2af, 24ae6636b, adbe59d1d (small).
  * server-context.cpp: adbe59d1d (58), 241372ae5 (307), 986036549 (34).
  * ggml-cpu.c / mmvq.cu: 098f547aa (fusion - the big follow-up), 107b9d596,
    512b2fb6d.
  * common.cpp / test-arg-parser.cpp: 18b913e29, 4120695e0.
- After each commit: compile the affected targets (CPU build). After every 4-5 commits:
  run the CPU test suite. Never proceed with a broken build.
- Final phase: full CPU build + ctest + doc/README merge check + diff review of the
  branch tip vs our branch (semantic diff of the 28 files) to catch porting drift.

## 5. Verification plan (what can be done on this machine)

Here (no GPU, no nvcc):
- CPU-only build: cmake -B build -DGGML_CUDA=OFF -DLLAMA_BUILD_TESTS=ON (verify exact
  preset/options at execution time; 16 cores -> -j16). Builds common, server, llama-bench,
  fit-params and all tests incl. test-moe-cache.cpp (compiles; skips at runtime without
  CUDA) and test-moe-cache-fit.cpp (runs).
- Run: ctest (test-moe-cache-fit, test-arg-parser, test-backend-ops CPU, test-grammar,
  test-sampling, plus the rest of the CPU suite) as regression check.
- Smoke: llama-cli/llama-bench --help, llama-server --help, fit-params --help
  (arg-parsing sanity), maybe a tiny-model CPU decode if a small GGUF is obtainable
  (optional; disk/network permitting).
- Static checks: git diff --check (whitespace), no non-ASCII in changed lines,
  clang-format on touched files per repo style.

Recommended but needs approval (disk usage ~3-4 GB, 17 GB free; no GPU needed to compile):
- Install CUDA toolkit (apt cuda-toolkit or nvidia-cuda-toolkit) and configure
  -DGGML_CUDA=ON for a compile-only build. This is the ONLY way to compile-verify
  moe-cache.cu against master's CUDA internals - the single biggest untested risk.
  nvcc build of moe-cache.cu (3145 lines) is slow but one-time. Without it, the CUDA
  path is verified by review only.

^ [User Comment]: Yes, this is fine to do!

User hardware validation (after completion, per branch docs):
- test-moe-cache (CUDA): full lifecycle/concurrency/invalidation suite on a real GPU.
- test-moe-cache-fit: CPU planner (no GPU needed).
- Real-model runs: llama-bench 3 arms (--moe-cache off --repack on / off --repack off /
  --moe-cache on --repack off) with -v, verify trace shows "[moe-cache] enabled" +
  nonzero hits; perplexity/logits comparison cache off vs on; coherent generation;
  retrieval at long context (DeepSeek V4 Flash DSpark 4-GPU profile and GLM-5.2 MTP
  3-GPU profile are documented in docs/backend/CUDA-MOE-CACHE.md).
- Server: target+draft (DSpark) and MTP profiles from the docs.

New tests: the branch already ships comprehensive tests (test-moe-cache.cpp 3154 lines,
test-moe-cache-fit.cpp, arg-parser extensions). If the port needs extra coverage, extend
existing test files (AGENTS.md forbids new test files without maintainer approval).

## 6. Risks & open questions

1. mmvq.cu port is the highest technical risk (master reworked the launch machinery).
   Mitigation: do it as a focused manual port with the branch tip as behavioral reference;
   compile-verify via CUDA toolchain if approved.
2. server-context.cpp port (3 commits, 443 lines vs master's 1779 changed) - highest
   conflict volume. Mitigation: port features, not hunks; keep master's fit/batch_ext
   behavior; re-run server build + arg tests after.
3. Semantic drift risk: resolving conflicts differently than the branch may change
   behavior invisibly. Mitigation: after the full port, diff our branch against branch
   tip per-file for the 28 touched files and review every functional difference.
4. No CUDA compile on this machine by default -> the CUDA path may only be validated on
   user hardware. Decision needed: install CUDA toolkit for compile-only verification?
5. Branch commit history preserved via -x; if a follow-up proves impossible to apply
   cleanly on the merged base, prefer a faithful manual port over dropping it - the
   feature depends on the whole series (especially 098f547aa fusion and the server/fit
   commits).
6. The .gitignore local change stays uncommitted; do not stash it (would interfere with
   working-tree state); cherry-pick works on HEAD, so it is safe either way.

## 7. Out of scope / housekeeping

- No git push, no PR creation, no commit messages for the user (AGENTS.md: forbidden).
- No upstream modification; simlu fork only.
- Do not add new files under tests/ beyond what the branch already brings.
- Report progress/blockers; user does the hardware validation.
