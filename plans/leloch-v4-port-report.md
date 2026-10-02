# Port report: leloch moe-cache on merged master (leloch-v4)

Branch: leloch-v4 (from master 66e165592, the merge of the v3 work)
Contents: the 30 commits of leloch-v3 (29 feature + fixup 849013fac),
cherry-picked with -x, chronological, fd349cf6a (the v3 report doc) skipped.
Date: 2026-10-02
Machine: 16 cores, nvcc 13.4.92, NO GPU (CUDA compile-only verification)

## Summary

All 30 v3 commits are applied on leloch-v4 (31 commits incl. this report).
Upstream between the two bases (19e28a277..66e165592) is 76 revs (75
non-merge); the plan text said 72. Only ONE conflict was hit: the base
commit's tools/llama-bench/README.md. No fixup commits were needed on v4.
All gates pass: CPU ctest 60/60 (final and after the base pick), --help
greps find moe-cache in llama-server / llama-bench / llama-fit-params,
git diff --check clean, no non-ASCII introduced, full CUDA compile-only
build green with nvcc 13.4. CUDA runtime is NOT verified (no GPU here).
Nothing was pushed and no PR was opened.

Branch diff vs master: 28 files changed, +9178/-413.

## Commits applied (30/30)

Every commit keeps its original message and provenance lines (v4 carries
both the upstream-reference pick line and the v3 pick line). New SHA -> v3
SHA:

  646310f48 -> 191f18fda  cuda: rework MoE expert cache execution (base)
  7a041f6ff -> bcfad579c  cuda: fix MoE cache capacity accounting
  5e992bc5e -> 66b500ea0  server: fix auto-selected DSpark device
  8392064e9 -> 2deb06f68  docs: update MoE cache validation
  716f69b1c -> 1b20e3e44  common: preserve implicit MoE cache provider settings
  d2a9fbda2 -> 72f06bae8  server: harden shared draft device placement
  dfa4c0deb -> 593c9cebb  cuda: fix MoE cache scratch budget replacement
  0d74474f8 -> 22246f7f2  llama: skip dormant MoE cache sessions
  4fcfa1799 -> c64cbddd9  cuda: bound MoE MMV tail-row reads
  e71f42235 -> 1d569e16b  cuda: tune MoE cache CPU overlap automatically
  ce9b65011 -> 6e504910c  cuda: adapt MoE cache admission to device capability
  849af2e1d -> 78ed2a072  cuda: align MoE cache pool allocation with fit
  e550b5d57 -> 7d2da1cd7  cuda: prefer generic MMV for compatible MoE cache nodes
  758e9b1ac -> f0701ce14  cuda: parallelize MoE cache fills on capable multi-GPU systems
  70cc882a7 -> 5dbf16646  server: account for MTP placement in fit reservations
  05280be2b -> 835e23adb  docs: update MoE cache validation
  799701aa5 -> 9bd089037  cuda: fuse cached MoE SwiGLU rows
  3ef6cad9f -> c637087ad  cuda: aggregate small MoE tensors into cache pools
  884dfea3d -> 4ef4d34bb  cuda: bound automatic MoE cache admission
  8358e9f36 -> c4241f76f  cuda: reject undersized automatic MoE cache slabs
  6e9ca7b61 -> e65e59874  cuda: enable multi-token MoE cache in forced modes
  c05b0d0b3 -> f2ad9ea52  common: surface MoE cache activation
  691db65dd -> bd44efe53  docs: fix MoE cache benchmark verbosity
  207b45725 -> e142b07d3  cuda: accelerate complete MoE cache pools
  8f59d621b -> a1c050549  cuda: report oversized MoE cache nodes
  c45db1d23 -> d0c3fb620  cuda: clarify MoE cache session diagnostics
  3f3a6ad06 -> d2242f1e0  docs: record MoE cache workload convergence
  3be05692b -> 7b587e238  cuda: report MoE cache pair residency
  6ef77e866 -> e5fad5f95  cuda: pack MoE cache dispatch inputs
  a2de2c860 -> 849013fac  server: drop redundant draft/MTP fit reservation (fixup)

## Conflict resolution (the only one)

Base commit 646310f48 (pick of 191f18fda): tools/llama-bench/README.md.
Upstream's docs fix 79625e056 (llama-bench : fix docs, #29464) landed inside
the region v3 rewrites. Resolution: keep v3's README rewrite as the base,
merge upstream's deltas on top -- the deprecated -mmp/-dio lines stay
removed, --load-mode default is auto -- and keep the 3-arm moe-cache/repack
benchmark procedure text from v3. Final-state check: v4 README equals master
README plus exactly the v3 feature additions.

All other 29 picks applied clean. Hard gate after the base pick (clean CPU
build + ctest 60/60) passed before continuing; suite re-run green after the
full pick series.

## Drift review (v3 vs v4, accepted drift)

Semantic review over the 28 feature files (diff 19e28a277..leloch-v3 minus
the v3 report doc) PASSED with zero unexplained diffs:

- 12 of 28 files differ between v3 and v4 (all explained by upstream's 76
  revs between bases); 16 are identical.
- Per file, diff(leloch-v3, leloch-v4) is content-identical to the upstream
  base diff diff(19e28a277, master) modulo hunk-offset/index bookkeeping;
  a patch-vs-patch check confirms every +/- feature line is preserved
  across branches (context-only deltas).
- 11 of the 12 drifted files match upstream byte-for-byte; the sole partial
  is tools/llama-bench/README.md, the documented conflict above.

Accepted drift notes:
- The mmvq.cu fusion machinery on the new base (MUSA guard rework 272aad8b9,
  sm70 table 42d958167) compiles cleanly with the branch's moe_launch
  threading; no re-expression was needed this round (the v3-era
  ggml_cuda_mm_fusion_args_device gotcha did not recur).
- fit.cpp / server-context.cpp: master's common_fit_extra_model path is
  unchanged; the fixup 849013fac applies semantically the same on v4.

## Verification results

CPU (build-cpu, GGML_CUDA=OFF, tests ON, Release):
- clean rebuild exit 0; ctest 60/60 passed, 0 failed (matches v3 baseline),
  at branch head a2de2c860.
- moe-cache flags present in llama-server, llama-bench and llama-fit-params
  --help.
- git diff --check master..leloch-v4 clean; no non-ASCII added by the
  branch (2 diff hits are pre-existing '+/-' in llama-bench README context,
  identical in master).

CUDA (build-cuda, GGML_CUDA=ON, compile-only, nvcc V13.4.92 via
/usr/local/cuda/bin):
- configure exit 0; full cmake --build -j16 exit 0; all targets built;
  libggml-cuda.so produced. moe-cache.cu, mmvq.cu, ggml-cuda.cu all
  recompiled; no errors touching them. Only pre-existing warnings.

## Known remaining risk

CUDA runtime is UNVERIFIED: this machine has no GPU, so moe-cache.cu,
mmvq.cu and the server CUDA paths are compile- and review-verified only.
Hardware validation below is mandatory before any performance claim.

## Hardware validation handoff

Per docs/backend/CUDA-MOE-CACHE.md. Record GPU model(s), PCIe topology, CPU,
memory, quantization, exact placement, build revision, env overrides.

1. test-moe-cache on a real GPU (lifecycle / concurrency / invalidation),
   plus the server two-slot concurrent test and test-moe-cache-fit.

2. llama-bench, 3 arms with -v (identical placement/ctx/batch/threads
   otherwise):
       arm 1: --moe-cache off --repack on
       arm 2: --moe-cache off --repack off
       arm 3: --moe-cache on --repack off
   Valid only when the trace shows "[moe-cache] enabled", the expected
   pools, and nonzero hits; use enough decode warmup (cold cache otherwise).

3. Perplexity/logits cache off vs on: expect statistical equivalence, NOT
   bit-identical output; coherent generation; long-context retrieval.

4. Server profiles: DSpark target+draft; GLM-5.2 embedded MTP 3-GPU
   (ctx 65536); DeepSeek V4 4-GPU (target-only, target+draft, sleep/wake,
   explicit --spec-draft-device). Verify output correctness, draft
   acceptance, no OOM/trim/fallback, GPUs idle at shutdown.

5. Optional: validate on new upstream models added between bases (Qwen4Exp
   MTP, GLM-5.3-Flash).

## Rollback

Nothing was pushed and no PR exists. master is untouched; leloch-v3 and the
full provenance chain remain local. The port is reproducible from this
report and the -x lines.
