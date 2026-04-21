# Notes: integrating `20a7cec991b83dace860bfc32e57eb59d65a6ad3` (v0.12 Jetson) into `main`

This document summarizes how the cherry-pick of `20a7cec99` (open source v0.12-jetson) was merged into current `main`, what was resolved automatically, and what may still need manual follow-up.

## CUDA 13.x vs sm\_87 (Jetson Orin)

Jetson Orin uses **compute capability 8.7** (`sm_87`). In this tree, support is expressed with the same CUDA toolchain options used for other Ampere-class targets, for example:

- CMake: `CMAKE_CUDA_ARCHITECTURES` / `filter_source_cuda_architectures` lists include **87** alongside existing arches (e.g. 80, 86, 89, 90).
- NVCC: `cpp/kernels/xqa/gen_cubins.py` uses `arch_options` including **87**; the generator builds with `-arch=compute_87 -code=sm_87` (same pattern as 80/86/90).

Nothing in these paths requires **CUDA 13** as a minimum: `sm_87` has been supported for a long time (CUDA 11.x era and later). If you add new device code, avoid introducing **CUDA 13-only** language or PTX features unless you intentionally raise the minimum toolkit.

## Conflicts resolved during merge

| Area | Resolution |
|------|------------|
| `cpp/include/tensorrt_llm/executor/executor.h` | Kept current `main` Executor API (`BufferView`, optional managed weights). **Added** `ExecutorConfig::use_engine_mmap` / `getUseEngineMmap()` (Jetson mmap load path); dropped legacy Jetson-only `Executor` constructors that took mmap at the `Executor` level. |
| `cpp/tensorrt_llm/nanobind/executor/executor.h` | Kept nanobind `Executor` bindings (Jetson branch targeted removed `pybind`). |
| `cpp/tensorrt_llm/pybind/executor/executor.cpp` | **Removed** — `pybind` executor was deleted on `main`; nanobind replaces it. |
| `tensorrt_llm/auto_parallel/cluster_info.py` | **Removed** — file was deleted on `main`; Jetson edits were not ported. |
| `contextFusedMultiHeadAttention/CMakeLists.txt` | Kept `main` object-library + `filter_source_cuda_architectures` / `foreach` pattern; **added arch 87** to both. |
| `fmhaRunner.cpp` | Merged SM checks: **includes `kSM_87`** alongside `kSM_80/86/89/90/100/103/120/121` from `main`. |
| `cubin/fmha_cubin.h` | **Not checked in** — still listed in `.gitignore`; `scripts/build_wheel.py` + `cpp/kernels/fmha_v2/setup.py` generate it. |
| `decoderMaskedMultiheadAttention/cubin/xqa_kernel_cubin.h` | **Restored `main`’s thin header** (extern declarations). |
| `decoderMaskedMultiheadAttention/CMakeLists.txt` | **Added 87** to `filter_source_cuda_architectures` ARCHS. |
| `selectiveScan/CMakeLists.txt` | Kept `main` layout (no legacy `filter_cuda_archs`). |
| `bufferManager.h` / Jetson memory pool helpers | Kept `main` header; **ported behavior** into `bufferManager.cpp` (`memoryPoolFree` clamps when used > reserved, e.g. Jetson). |
| `tllmRuntime.cpp` | Merged **GDS**, **mmap** (`RawEngine::useMMap()`), and **StreamReader** paths; POSIX mmap implementation is **Windows-stubbed**. |
| `cpp/include/tensorrt_llm/runtime/rawEngine.h` | Kept constructor + `useMMap()` from cherry-pick (already aligned with `main` merge). |
| `setup.py` / `scripts/build_wheel.py` | Kept `main` constraints + venv flow; **select `requirements-jetson.txt` / `requirements-dev-jetson.txt`** when `tegra` + `aarch64`. |
| `tensorrt_llm/profiler.py` | NVML v2 default; **on Jetson**, skip NVML-based device memory in `device_memory_info` (returns zeros + one-time warning) when pynvml/driver is unsuitable. |
| `examples/run.py`, `summarize.py` | **`--use_mmap`** → `ModelRunnerCpp.from_dir`; **`summarize.py`** skips `mpi_broadcast` on Tegra aarch64 (single-process Jetson). |
| `tensorrt_llm/runtime/model_runner_cpp.py` | **`use_mmap`** sets `trtllm_config.use_engine_mmap` for C++ engine load. |

## Manual follow-up (recommended)

1. **FMHA cubin header and registration**  
   After adding `sm_87` cubin sources, run the normal wheel / FMHA generation flow so `cubin/fmha_cubin.h` (generated, gitignored) matches the committed `*_sm87.cubin.cpp` blobs.

2. **XQA meta table (`xqa_kernel_cubin.cpp`)**  
   On `main`, `xqa_kernel_cubin.cpp` is a large generated/LFS artifact. Adding `87` to `cpp/kernels/xqa/gen_cubins.py` is done; **re-run** the XQA cubin generator and merge the updated `xqa_kernel_cubin.cpp` so `sXqaKernelMetaInfo` includes the new `*_sm_87` translation units.

3. **Static import libraries for L4T**  
   The cherry-pick adds `aarch64-l4t-gnu/*.a` (LFS). Confirm they match your TensorRT / JetPack version and CI policy.

## Reference

- Original commit: `20a7cec991b83dace860bfc32e57eb59d65a6ad3` (message: `open source v0.12-jetson`).
