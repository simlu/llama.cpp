# Implementation report: leloch/moe-cache-v2-pr port onto simlu/llama.cpp master

Branch: leloch-v3 (from origin/master 19e28a277)
Reference source: leloch/moe-cache-v2-pr tip e3096b046 (merge-base 15586e2d7, 29 linear commits)
Date: 2026-09-30
Machine: 16 cores, nvcc 12.4.131, NO GPU (CUDA compile-only verification)

## Summary

The complete 29-commit moe-cache-v2 series was ported onto current master and is
committed on branch leloch-v3 (30 commits total = 29 cherry-picks with -x
provenance + 1 local fixup from the Step 4 drift review). CPU build and full
test suite pass (60/60 in the final run; the two baseline environmental
failures - LFS vocab pointers and missing jinja2 - were fixed by installing
git-lfs/python3-jinja2 and pulling LFS objects). CUDA compile-only build passes
with all targets (nvcc 12.4, GGML_CUDA=ON). CUDA runtime is NOT verified: this
machine has no GPU. Nothing was pushed and no PR was opened.

## Commits applied (29/29 reference commits + 1 fixup)

Chronological order verified against `leloch/moe-cache-v2-pr --not origin/master`
(subject list and SHA order match 1:1). Every commit carries a
`(cherry picked from commit ...)` provenance line. New SHA -> source SHA:

Phase A (base commit, the big one, +6814/-279):
  191f18fda -> f2d7f9303  cuda: rework MoE expert cache execution

Phase B (28 follow-ups, all strictly chronological):
  bcfad579c -> bb1ea6933  cuda: fix MoE cache capacity accounting
  66b500ea0 -> 986036549  server: fix auto-selected DSpark device
  2deb06f68 -> 6d20f3770  docs: update MoE cache validation
  1b20e3e44 -> 4120695e0  common: preserve implicit MoE cache provider settings
  72f06bae8 -> 241372ae5  server: harden shared draft device placement
  593c9cebb -> f04585925  cuda: fix MoE cache scratch budget replacement
  22246f7f2 -> b7024cc10  llama: skip dormant MoE cache sessions
  c64cbddd9 -> 512b2fb6d  cuda: bound MoE MMV tail-row reads
  1d569e16b -> f0bd8b1b5  cuda: tune MoE cache CPU overlap automatically
  6e504910c -> 24ae6636b  cuda: adapt MoE cache admission to device capability
  78ed2a072 -> 43c3be2af  cuda: align MoE cache pool allocation with fit
  7d2da1cd7 -> d5ffc999f  cuda: prefer generic MMV for compatible MoE cache nodes
  f0701ce14 -> 135e204d2  cuda: parallelize MoE cache fills on capable multi-GPU systems
  5dbf16646 -> adbe59d1d  server: account for MTP placement in fit reservations
  835e23adb -> a38ba195a  docs: update MoE cache validation
  9bd089037 -> 098f547aa  cuda: fuse cached MoE SwiGLU rows
  c637087ad -> 32fbcd681  cuda: aggregate small MoE tensors into cache pools
  4ef4d34bb -> c0d7d916b  cuda: bound automatic MoE cache admission
  c4241f76f -> a68d9dad6  cuda: reject undersized automatic MoE cache slabs
  e65e59874 -> 9aeef2cfd  cuda: enable multi-token MoE cache in forced modes
  f2ad9ea52 -> 18b913e29  common: surface MoE cache activation
  bd44efe53 -> 7d4af9e6e  docs: fix MoE cache benchmark verbosity
  e142b07d3 -> b07c53689  cuda: accelerate complete MoE cache pools
  a1c050549 -> 107b9d596  cuda: report oversized MoE cache nodes
  d0c3fb620 -> 78104ec46  cuda: clarify MoE cache session diagnostics
  d2242f1e0 -> fd612c058  docs: record MoE cache workload convergence
  7b587e238 -> f31e9b7c7  cuda: report MoE cache pair residency
  e5fad5f95 -> e3096b046  cuda: pack MoE cache dispatch inputs

Local fixup (authored in Step 4 drift review, NOT part of the reference series):
  849013fac -> server: drop redundant draft/MTP fit reservation (extra_model
  supersedes it) -- removes the old reserve-before-fit block in
  tools/server/server-context.cpp that double-counted draft/MTP memory against
  the fit budget (master's common_fit_extra_model already accounts for it), and
  restores the n_threads/cpu_mask empty-defaulting in tools/llama-bench/
  llama-bench.cpp (lost during the Phase A 3-way merge; without it the
  benchmark loop iterated zero configurations when -t/-C were not passed).
  Diff: tools/server/server-context.cpp -129/+7, tools/llama-bench/llama-bench.cpp +6.

Note: `git rev-list --count origin/master..leloch-v3` reports 30, not 29: the
29 reference commits are all applied (each verified with provenance lines);
the 30th is the local fixup above. Total branch diff vs origin/master:
28 files changed, +9180/-413.

## Conflicts resolved (per file)

### Phase A (cherry-pick f2d7f9303): 9 conflict regions, 10 files auto-merged

Policy: keep branch semantics, preserve master's evolved structure; additive
changes keep both sides; heavily-diverged files were 3-way merged with
`git merge-file` (branch as current, merge-base 15586e2d7 as base, master HEAD
as other).

- include/llama.h, common/common.h, src/llama-ext.h, src/llama-cparams.h,
  ggml/src/ggml-backend-reg.cpp, ggml/src/ggml-backend.cpp, src/llama-context.cpp,
  src/llama-model.cpp, common/common.cpp, ggml/src/ggml-cuda/ggml-cuda.cu:
  auto-merged (additive hunks; branch hooks land on master's evolved code).
- common/arg.cpp: additive - kept BOTH master's `-ncffn` and branch's
  `--moe-cache`; branch option wrapped in add_opt(common_arg(...)) to match
  surrounding style.
- tests/CMakeLists.txt: dropped branch's duplicate
  `llama_build_and_test(test-backend-ops.cpp)` (master defines it via
  `llama_build`); kept branch's new moe-cache test targets.
- tests/test-arg-parser.cpp: removed markers; restored includes <cmath>,
  <limits>, <cstdlib> that a naive marker deletion had removed.
- tools/fit-params/fit-params.cpp: merged call passes BOTH master's `nullptr`
  and branch's `&params.moe_cache`.
- tools/llama-bench/llama-bench.cpp (3-way merge, 3 conflicts): kept branch's
  string-based `repack` (mapped to `mparams.use_extra_bufts`) and removed all
  of master's vector-based repack machinery (member, default, parse block,
  rpk loop/initializers, display name, field width, duplicate ctor/value
  assignments); kept master's `lazy_mode`, `llm_ffn_block_regex` rename,
  `hf_file` fix, `nullptr` fit arg; added the missing
  `/* fit_params_target */ { 0 },` default. Fields/values counts in get_map
  match (45 each).
- tools/llama-bench/README.md: branch's rewritten README taken as base
  (master's samples were stale, referencing removed fields); ported master's
  non-superseded deltas (`auto` load-mode, `-lzm/--lazy-mode` line,
  `(DEPRECATED IN FAVOUR OF --load-mode)` notes).
- common/fit.cpp (3-way merge, 4 conflicts): impl signature and caller carry
  BOTH branch's `moe_cache`/`moe_tensors` AND master's `extra` model params
  (`n_streams`, `n_ctx_auto`, `add_extra_memory`); includes keep both sides
  (common.h, json.h); body splice = master's structure + branch's
  `&moe_tensors` argument.
- common/fit.h: auto-merged with both parameter sets.
- ggml/src/ggml-cuda/mmvq.cu (3-way merge, 5 conflicts): master's fusion
  launch machinery (`halve_iters`, `mul_mat_vec_q_switch_fusion` templates,
  `calc_launch_params`) merged with branch's `act_ids` threading in the kernel
  signature, `moe_launch`, and both call sites in `switch_ncols_dst`; one call
  site required `ggml_cuda_mm_fusion_args_device{}` instead of `nullptr`
  (caught by the CUDA compile during the gate).
- tools/server/server-context.cpp: master's `common_fit_extra_model` path
  supersedes the branch's old spec pre-reserve block; kept master's structure
  and comment. (This exact leftover block was later identified as the
  double-counting drift bug and removed in 849013fac.)

### Phase B: manual conflict ports (branch semantics + master's evolved APIs)

- 241372ae5 -> 72f06bae8 (tools/server/server-context.cpp): branch's
  `params_load` pristine-refit semantic with master's `server_output_limits` API.
- b7024cc10 -> 22246f7f2 (src/llama-context.cpp): kept both master's
  `llama_graph_n_input_tensors` and branch's new dormant-session skip function.
- 512b2fb6d -> c64cbddd9 (ggml/src/ggml-cuda/mmvq.cu): branch tail-row bound
  with the gate read wrapped inside.
- 098f547aa -> 9bd089037 (ggml/src/ggml-cuda/mmvq.cu, 10 conflict regions):
  branch's fused-kernel design; master's prefetch helpers
  (`mmvq_should_prefetch`/`mmvq_prefetch_l2`) kept; leftover `use_gate` block
  removed. After this port, moe-cache.cu, mmvq.cuh and ggml-backend-moe-cache.h
  are byte-identical to the branch tip.

All other Phase B commits applied cleanly (docs/tests/new-file group).

## Verification results

CPU build (fresh):
- cmake -B build-cpu -DGGML_CUDA=OFF -DLLAMA_BUILD_TESTS=ON
  -DCMAKE_BUILD_TYPE=Release; cmake --build build-cpu -j16: clean, exit 0.
- ctest --test-dir build-cpu: 60/60 PASS in the final Step 4 run (the two
  earlier environmental failures - test-tokenizers-ggml-vocabs LFS pointers and
  test-jinja-py missing jinja2 - were fixed by installing git-lfs and
  python3-jinja2 and pulling the LFS objects; they were baseline failures,
  not port failures).
- Targeted re-run (this report): test-arg-parser (#40), test-moe-cache (#43),
  test-moe-cache-fit (#44) all PASS.

CUDA compile (no GPU; compile-only):
- cmake -B build-cuda -DGGML_CUDA=ON -DLLAMA_BUILD_TESTS=ON
  -DCMAKE_BUILD_TYPE=Release; cmake --build build-cuda -j16: exit 0, all
  targets built (llama-app, llama-server, llama-bench, llama-batched-bench,
  test-moe-cache, test-moe-cache-fit).
- All three key CUDA sources were recompiled in that run (moe-cache.cu,
  mmvq.cu, ggml-cuda.cu); object timestamps confirmed newer than sources.

Smoke greps (arg parsing):
- build-cpu/bin/llama-bench --help | grep -i moe-cache ->
  "--moe-cache <auto|on|off|0|MiB>  (default: auto)"
- build-cpu/bin/llama-server --help | grep -i moe-cache ->
  "--moe-cache MODE  adaptively cache the hottest CPU-resident MoE experts in spare VRAM"
- build-cpu/bin/llama-fit-params --help | grep -i moe-cache -> present
  (the binary is llama-fit-params in this tree, not fit-params).

Static checks:
- git diff --check origin/master..leloch-v3: CLEAN.
- ASCII: branch diff vs origin/master contains exactly 2 non-ASCII bytes
  (2x "+/-" in a llama-bench README benchmark table), inherited verbatim from
  the reference branch's README rewrite; the drift diff vs the reference tip
  is ASCII-clean. No authored content violates the ASCII rule.
- clang-format-22 via git-clang-format: violation counts identical in scale to
  the branch tip itself (~11.1k lines both, per-file delta +-48); the repo has
  no repo-wide clang-format gate, so no reformatting was applied (it would
  create drift vs the reference branch).

Semantic drift review (Step 4, 28 files, 3-way faithfulness reconstruction
against the reference tip):
- 18 files exact/faithful (byte-identical or context-only differences).
- 8 files accepted after review (conflict-marker-only remnants or master
  evolution that is correct on top of the port).
- 2 real drift bugs found and FIXED in 849013fac:
  1. server-context.cpp kept the old reserve-before-fit block while master's
     common_fit_extra_model path is active -> draft/MTP memory double-counted
     in the fit budget. Old block removed; shared-draft placement
     (make_params_dft) is still used for draft context creation.
  2. llama-bench.cpp lost the n_threads/cpu_mask empty-defaulting -> zero
     benchmark iterations when -t/-C were not passed. Defaults restored from
     cmd_params_defaults before the benchmark loop.

## Known remaining risk

CUDA runtime is UNVERIFIED. This machine has no GPU, so moe-cache.cu, mmvq.cu
and the server CUDA paths are compile-verified and review-verified only. The
cache is designed to fall back to the stock CPU path on any failure, so a
runtime defect should degrade performance rather than crash, but that is an
assumption, not a result. Hardware validation below is mandatory before any
performance claim.

## Hardware validation handoff

Per docs/backend/CUDA-MOE-CACHE.md. All four items are required; results must
be recorded with GPU model(s), PCIe topology, CPU model, memory channels and
speed, model quantization, exact placement, build revision, and environment
overrides.

1. test-moe-cache on a real GPU (lifecycle / concurrency / invalidation).
   A CUDA build can exercise the synthetic success and failure paths with:
       CUDA_VISIBLE_DEVICES=0 ./build/bin/test-moe-cache
   Also run the server two-slot concurrent test and the model-free
   concurrent-node/nested-session coverage (test-moe-cache-fit).

2. llama-bench, 3 arms with -v (identical placement/context/batch/threads/
   warmup otherwise):
       arm 1: --moe-cache off --repack on   (optimized CPU-expert baseline)
       arm 2: --moe-cache off --repack off  (canonical CPU-expert baseline)
       arm 3: --moe-cache on --repack off   (cache arm; or a fixed MiB budget)
   Arm 1 vs arm 3 measures the end-to-end configuration benefit; arm 2 vs arm 3
   isolates cache execution from repacking. A result is valid only when the
   trace log (pass -v, and -lv 4 for backend diagnostics) contains
   "[moe-cache] enabled", the expected pools, and nonzero hits in periodic or
   teardown statistics. Use enough decode warmup (pool creation waits for
   graph-shape discovery; admission needs repeated demand; fills are async) -
   a one-token warmup measures a cold cache. Note: other common-parser
   programs spell the switches --repack/--no-repack.

3. Quality equivalence: perplexity or logits with cache off vs on (expect
   statistical equivalence, NOT bit-identical output - CPU and CUDA rounding
   can change near-tie token decisions), coherent generation, and a
   long-context retrieval test (e.g. the 54k-token DeepSeek V4 and 58k-token
   GLM-5.2 archival-prompt patterns from the docs).

4. Server profiles (from the docs, adjust paths/thread counts to the host):
   - DSpark target+draft:
       CUDA_VISIBLE_DEVICES=0,1,2,3 ./build/bin/llama-server -m <target.gguf>
         -md <draft.gguf> --spec-type draft-dspark --spec-draft-n-max 5
         --fit on --moe-cache auto -t 24 -tb 48 -fa on -c 8192 -np 1
         -ctk q8_0 -ctv q8_0
   - GLM-5.2 embedded MTP, 3 GPUs, context 65536:
       CUDA_VISIBLE_DEVICES=0,1,2 GGML_CUDA_MOE_CACHE_RESERVE_MB=3072 \
         ./build/bin/llama-server -m <GLM-5.2-UD-IQ2_M-*.gguf> --fit on
         --moe-cache auto -c 65536 -np 1 -b 4096 -ub 512 -t 44 -tb 48 -fa on
         --load-mode mmap --cache-ram 0 --spec-type draft-mtp
         --spec-draft-n-max 3 --spec-draft-p-min 0.5
   - DeepSeek V4 4-GPU: same DSpark profile with the DeepSeek-V4-Flash model
     and the draft sidecar; verify target-only and target+draft modes, plus
     sleep/wake reload and an explicit --spec-draft-device path.
   Verify: correct output (exact `1..200` style checks where applicable),
   draft acceptance rate, no OOM/trim/fallback errors, and shutdown returning
   all GPUs to idle memory.

Environment overrides worth knowing (implementation controls, not a stable
CLI): GGML_CUDA_MOE_CACHE_RESERVE_MB (3072 default),
GGML_CUDA_MOE_CACHE_MIN_EXPERT_KB (512 on CC 8.0+, 1024 otherwise),
GGML_CUDA_MOE_CACHE_MAX_BATCH (8), GGML_CUDA_MOE_CACHE_STATS (N prints
periodic counters), GGML_CUDA_MOE_CACHE_FAIL (fallback fault injection for
testing only), GGML_CUDA_MOE_CACHE_SERIAL_FILL,
GGML_CUDA_MOE_CACHE_MIN_CC, GGML_CUDA_MOE_CACHE_NDEV. Keep
GGML_OP_OFFLOAD_MIN_BATCH above the decode batch size or generic offload can
bypass the cache.

## Rollback

Nothing was pushed and no PR exists. master is untouched; the reference branch
and all provenance lines remain in the local repository. The branch state is
fully described by this report, and each commit retains its provenance line, so
the port can be reproduced or inspected at any time.
