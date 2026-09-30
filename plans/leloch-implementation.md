# Implementation plan: port leloch/moe-cache-v2-pr onto simlu/llama.cpp master

Derived from plans/leloch.md (all facts re-verified against the live repo on 2026-09-30):

- leloch/moe-cache-v2-pr tip = e3096b046, merge-base with origin/master = 15586e2d7, 29 linear commits
- origin/master = 19e28a277; local HEAD = b1739ee75 (plan + .gitignore committed locally)
- Machine: 16 cores, 17 GB free disk, NO nvcc/GPU

Branch: `leloch-v3`, created from origin/master. Every cherry-pick uses `-x`.
Never push, never PR, never write commit messages for the user (AGENTS.md).
The .gitignore/plan commits live on local master only; branching from origin/master keeps
them out of leloch-v3 automatically - no action needed.

---

## Step 0 - Preflight (5 min)

    git fetch origin && git fetch leloch
    git rev-parse leloch/moe-cache-v2-pr        # expect e3096b046...
    git merge-base leloch/moe-cache-v2-pr origin/master   # expect 15586e2d7...
    git rev-list --count leloch/moe-cache-v2-pr --not origin/master  # expect 29
    git status                                    # expect clean (or only local commits on master)
    git checkout -b leloch-v3 origin/master

Gate: branch exists at 19e28a277, working tree clean. Abort here if any fact drifted.

## Step 1 - Install CUDA toolkit (background, user-approved)

    sudo apt-get update && sudo apt-get install -y nvidia-cuda-toolkit   # ~3-4 GB
    nvcc --version

Run this early in the background - it is slow and is the only way to compile-verify
moe-cache.cu against master's CUDA internals. Gate before Step 5: nvcc present.
(If the distro package is too old, use NVIDIA's cuda-toolkit repo instead.)

## Step 2 - Phase A: base commit f2d7f9303 (the big one, +6814/-279)

    git cherry-pick -x f2d7f9303

Expected conflicts (18 files). Resolution order and policy:

2a. Mechanical (low risk, resolve first):
    include/llama.h, common/common.h, common/arg.cpp, src/llama-ext.h,
    src/llama-cparams.h, tools/fit-params/fit-params.cpp, tests/CMakeLists.txt,
    tests/test-arg-parser.cpp, ggml/src/ggml-backend-reg.cpp
    Policy: keep branch semantics, take master structure; these are additive hunks.

2b. Integration (medium): ggml/src/ggml-backend.cpp, src/llama-context.cpp,
    src/llama-model.cpp, common/common.cpp, ggml/src/ggml-cuda/ggml-cuda.cu
    Policy: branch's hooks (invalidation on write paths, set_moe_cache call sites,
    llama_model_get_moe_tensor_info, cparams mapping, alloc-trim hook) land on top of
    master's evolved code; verify each master-side change nearby is preserved
    (batch_ext, W4A4/prec_policy, CUDA graph MTP, multi-output sampling in
    llama-context.cpp; look_ahead_size/cuMemCreate in ggml-cuda.cu).

2c. fit.cpp / fit.h (MED-HIGH): keep master's n_streams ctx logic, LOG_JSON, auto-fit
    max-ctx revert (#29437). Insert branch's common_moe_cache_plan_fit + integration
    into common_params_fit_impl + the common_fit_params() signature change, then update
    the 3 master call sites (verify at time: common/common.cpp, tools/fit-params/
    fit-params.cpp, tools/llama-bench/llama-bench.cpp).

2d. mmvq.cu (HIGHEST RISK - manual port, not hunk apply):
    Master reworked the launch machinery (mmvq_parameter_table_id, mmvq_should_prefetch,
    halve_iters for GB10, small_k promotion, mul_mat_vec_q_moe<bool has_fusion>).
    Procedure:
      1. diff branch-tip mmvq.cu vs master mmvq.cu and enumerate the branch's deltas:
         act_ids_ptr threading, allow_small_k control, tail-row bounds, pool matvec
         kernels ggml_cuda_moe_cache_mmv / _fused.
      2. Re-express each delta on master's current function shapes. The branch tip is
         the behavioral reference; do not apply hunks mechanically.

2e. server-context.cpp (HIGH conflict volume): port branch's FEATURES only - shared
    draft device auto-placement, MTP/DRaft fit reservation accounting, DSpark device
    selection fix, hardened placement. Preserve master's batch_ext migration,
    ctx-per-slot, mmproj device fit, sleep refactor, yield_to_queue thread model.

    git cherry-pick --continue   (keeps original message + -x provenance line)

Gate (hard stop before Phase B):
    cmake -B build-cpu -DGGML_CUDA=OFF -DLLAMA_BUILD_TESTS=ON -DCMAKE_BUILD_TYPE=Release
    cmake --build build-cpu -j16
    ctest --test-dir build-cpu --output-on-failure
    CPU build + full CPU suite MUST pass.

## Step 3 - Phase B: follow-ups 2..29, strictly chronological

Each commit: `git cherry-pick -x <sha>`; after each, build the affected targets;
after every 4-5 commits run the CPU suite. Never proceed with a broken build.
If a cherry-pick fails, use `git cherry-pick --abort` and do a FAITHFUL MANUAL PORT
(same semantics) instead - do not drop any commit; the series is interdependent
(esp. 098f547aa fusion).

3a. Expect-clean group (new files / docs / tests only):
    bb1ea6933, 986036549, 6d20f3770, 4120695e0, f04585925, f0bd8b1b5, d5ffc999f,
    135e204d2, a38ba195a, 9aeef2cfd, 7d4af9e6e, 78104ec46, fd612c058, f31e9b7c7,
    e3096b046
    Note: commits 14 (135e204d2) and 29 (e3096b046) touch only moe-cache.cu + tests -
    they apply after moe-cache.cu exists in our tree (it does, post-Phase-A).

3b. Targeted conflicts (small, file-specific):
    - 512b2fb6d  mmvq.cu (3 lines, tail-row bounds) - port onto our ported mmvq.cu
    - 107b9d596  ggml-cpu.c + header + moe-cache.cu (oversized node reporting)
    - b7024cc10  llama-context.cpp (+49, dormant sessions skip)
    - 24ae6636b  fit.cpp/h + moe-cache.cu + llama-context.cpp (admission by capability)
    - 43c3be2af  fit.cpp/h + moe-cache.cu (pool alignment with fit)
    - a68d9dad6  fit.cpp/h + moe-cache.cu + llama-context.cpp (undersized slab reject)
    - 32fbcd681  fit.cpp (aggregate small tensors into pools)
    - adbe59d1d  fit.cpp/h + server-context.cpp (MTP placement reservations)
    - 986036549  server-context.cpp (DSpark device fix)  [listed in 3a only if clean]
    - 241372ae5  server-context.cpp (+307, hardened draft placement)
    - 18b913e29  common.cpp + moe-cache.cu (surface activation)
    - 4120695e0  common.cpp + arg-parser tests (implicit provider settings)
    Policy per file is the same as Step 2: keep branch semantics, preserve master's
    surrounding changes.

3c. 098f547aa - fused SwiGLU for resident rows (+1736, the biggest follow-up):
    Treat as a mini-Phase-A. Manual port of:
    - ggml-cpu.c: refactor into ..._impl() with row_mask + allow_moe_cache, graph-level
      gate/up/SwiGLU subgraph detection
    - mmvq.cu: fused pool kernel (_fused)
    - moe-cache.cu: +750 lines of fusion plumbing
    - tests: +655 lines
    Gate after it: full CPU build + ctest.

## Step 4 - Final CPU verification

    rm -rf build-cpu && cmake -B build-cpu -DGGML_CUDA=OFF -DLLAMA_BUILD_TESTS=ON -DCMAKE_BUILD_TYPE=Release
    cmake --build build-cpu -j16
    ctest --test-dir build-cpu --output-on-failure
    # smoke: arg-parsing sanity
    build-cpu/bin/llama-bench --help | grep -i moe-cache
    build-cpu/bin/llama-server --help | grep -i moe-cache
    build-cpu/bin/fit-params --help | grep -i moe-cache
    # static checks
    git diff --check leloch-v3
    git diff leloch-v3 | grep -P '[^\x00-\x7F]' | head   # expect empty (ASCII-only rule)
    clang-format on the 28 touched files (repo .clang-format)

Semantic drift check (catches conflict-resolution mistakes):
    git diff leloch/moe-cache-v2-pr...leloch-v3 -- <the 28 files>
    Review every functional difference; document accepted drift (context-only
    differences are fine, behavioral ones are not). Use the branch tip as reference
    for intent on each file.

## Step 5 - CUDA compile-only verification (after Step 1 toolkit)

    cmake -B build-cuda -DGGML_CUDA=ON -DLLAMA_BUILD_TESTS=ON -DCMAKE_BUILD_TYPE=Release
    cmake --build build-cuda -j16
    (LLAMA_CUBLAS is a deprecated alias for GGML_CUDA - do not use it.)

Compiles moe-cache.cu (3145 lines, slow but one-time), mmvq.cu, ggml-cuda.cu, server,
bench, and test-moe-cache*.cpp. This is the single biggest untested risk - the CUDA
path is otherwise review-only. No GPU is needed to compile; runtime stays unverified
until user hardware.

## Step 6 - Report and handoff

Artifacts: branch leloch-v3 (29/29 commits), build-cpu/, build-cuda/.
Report: commits applied, conflicts resolved + how, manual ports (mmvq.cu, fusion,
server-context), verification results, known remaining risk (CUDA runtime).

Hand to user for hardware validation (per docs/backend/CUDA-MOE-CACHE.md):
    1. test-moe-cache on a real GPU (lifecycle/concurrency/invalidation)
    2. llama-bench 3 arms: --moe-cache off --repack on | off --repack off |
       --moe-cache on --repack off, with -v; verify "[moe-cache] enabled" + nonzero hits
    3. Perplexity / logits cache off vs on (expect statistical equivalence, not
       bit-identical); long-context retrieval test
    4. Server profiles: DSpark target+draft, GLM-5.2 MTP 3-GPU, DeepSeek V4 4-GPU

## Rollback

Cherry-picks live on a branch: `git cherry-pick --abort` for the in-flight commit,
or `git reset --hard origin/master && git checkout -B leloch-v3 origin/master` to
start over. Nothing is pushed; master is untouched.

## Effort estimate

- Step 0: 5 min
- Step 1: 15-30 min (download/install, background)
- Step 2 (Phase A): 4-8 h (mmvq.cu + server-context.cpp dominate)
- Step 3 (Phase B): 2-4 h + compile time; 098f547aa ~1-2 h of that
- Step 4: 1-2 h (build + ctest + drift review)
- Step 5: 1-2 h (CUDA compile; nvcc is slow on moe-cache.cu)
- Total: ~9-16 h of focused work, mostly serialized on the two hard files.
