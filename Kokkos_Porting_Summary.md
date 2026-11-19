# AMReX on Kokkos: Porting and Performance Summary

- **Branch:** `topic-kokkos` (local only, not pushed), based on `development`
  at `ff7868ce94` ("CI: use Visual Studio's clang-cl in the MSVC Clang job
  (#6257)").
- **Commits:** `11bf34ce37` "Port the GPU portability layer to Kokkos" (code),
  `a6ce55fa9b` "Docs: Kokkos as the GPU portability layer" (Sphinx docs), and
  the commit that adds this report.
- **Date:** 2026-11-19.

## Summary

- GPU builds of AMReX now implement the on-node performance-portability layer
  with [Kokkos](https://kokkos.org) (>= 5.0, tested with 5.3.0). Kokkos is
  required for CUDA, HIP and SYCL builds, and optional for CPU builds.
- The user-facing AMReX API is unchanged, and none of AMReX's tests had to
  change.
- AMReX's own CUDA/HIP/SYCL implementations of what Kokkos provides are
  removed: element-wise kernel launches, `Gpu::Handler`, atomics, `ReduceOps`
  / `ReduceData`, block reductions, `AnyOf`, stream creation (CUDA/HIP), arena
  system allocations and host/device copies. The diff against `development`
  is 39 files, +2000 / −2447 lines (including 148 lines of documentation).
- **GPU performance** (NVIDIA RTX A2000, median of 5 runs): of 17 benchmark
  cases, 6 are faster (by 4% to 2x: `MultiFab` sum/norms/min/max, atomic
  max, `ParallelForRNG`), 9 are within 2% (stencils, flat reductions, scans,
  atomic add, FillBoundary; the slowest of these is an `MFIter` stencil loop
  at 1.016x), and 2 are slower: launches of very small kernels (1.24x, from
  Kokkos' host overhead of about 0.8 µs per launch) and a fused three-value
  `ReduceOps` over a `MultiFab` (1.11x). At least six alternatives were
  tried for each slower case (see below).
- **CPU performance** is unchanged: AMReX's CPU code paths are not
  dispatched through Kokkos, because Kokkos was slower there. A CPU build
  with `AMReX_KOKKOS=ON` measures within ±2% (noise) of `development`.
- **Tests:** all 136 CUDA tests, all 138 CPU+OpenMP tests, the 9 tests of
  the default CUDA configuration (relocatable device code on, Kokkos built by
  AMReX) and a GNU make CUDA build pass.
- **Not verified:** HIP and SYCL (no compilers or devices available; the
  code was written but never compiled), multi-rank MPI runs, and data-center
  GPUs.

## Goal

Port AMReX's portability layer to Kokkos, remove the functionality AMReX
re-implemented that Kokkos provides, keep AMReX's user-facing API, and keep
the default performance on CPU and GPU as good as or better than
`development`.

## Design

### Build system

- **CMake:** a new option `AMReX_KOKKOS` is `ON` by default and required when
  `AMReX_GPU_BACKEND` is `CUDA`, `HIP` or `SYCL`, and `OFF` by default for CPU
  builds. AMReX calls `find_package(Kokkos 5.0)`. If no Kokkos is found (or
  `AMReX_kokkos_internal=ON`), AMReX builds one with `FetchContent`
  (`AMReX_kokkos_src`, `AMReX_kokkos_repo`, `AMReX_kokkos_branch`, default
  tag `5.3.0`). The internal build matches AMReX's configuration: backend,
  `CMAKE_CUDA_ARCHITECTURES` → `Kokkos_ARCH_*`, `AMReX_AMD_ARCH`, the Intel
  architecture, OpenMP and relocatable device code (RDC).
- **Configuration checks** (CMake `FATAL_ERROR` with an explanation): the
  Kokkos backend must match AMReX's, `Kokkos_ENABLE_SERIAL` is needed,
  `Kokkos_ENABLE_OPENMP` if `AMReX_OMP=ON`, a CUDA Kokkos must be built with
  `Kokkos_ENABLE_COMPILE_AS_CMAKE_LANGUAGE=ON` (AMReX compiles its sources as
  the CMake `CUDA` language, not through `nvcc_wrapper`), and Kokkos' RDC
  mode must equal `AMReX_GPU_RDC`. `AMReX_Kokkos.H` repeats the RDC check at
  compile time, so GNU make users get a clear message instead of an error
  from inside Kokkos' atomics (desul).
- `AMREX_USE_KOKKOS` is in the generated `AMReX_Config.H`, and
  `AMReXConfig.cmake` calls `find_dependency(Kokkos)`.
- **GNU make:** `USE_KOKKOS` (default and required `TRUE` in GPU builds),
  `KOKKOS_HOME`, and the new `Tools/GNUMake/packages/Make.kokkos`.

### Initialization

- `amrex::Initialize` initializes Kokkos on the device that AMReX selected,
  with AMReX's OpenMP thread count, unless the application initialized
  Kokkos already. `amrex.kokkos_disable_warnings` (default `1`) silences
  Kokkos' start-up warnings.
- Kokkos cannot be initialized again after `Kokkos::finalize`, but AMReX can
  (`amrex::Finalize` followed by `amrex::Initialize`). AMReX therefore
  finalizes a Kokkos that it initialized at program exit (`std::atexit`).

### Streams

- AMReX's stream 0 is the default instance of
  `Kokkos::DefaultExecutionSpace`, so Kokkos kernels that an application
  launches on the default instance are ordered with AMReX's default stream.
- On CUDA and HIP, the other AMReX streams are instances from
  `Kokkos::Experimental::partition_space`. On SYCL, AMReX keeps creating its
  in-order queues and wraps them in `Kokkos::SYCL` instances. External
  streams (`Gpu::setExternalGpuStream`) are wrapped in non-owning instances.
- New API: `Gpu::kokkosExecSpace()` (instance of the current stream),
  `Gpu::KokkosExecSpace`, `Gpu::KokkosDeviceMemSpace`.
- `Gpu::streamSynchronize` fences the instance; `Gpu::synchronize` calls
  `Kokkos::fence`.

### Kernel launches

- `ParallelFor` (1D, `Box`, `Box` + components, the multi-box and `MultiFab`
  variants), `ParallelForRNG`, `launch(n)` and `launch(box)` run as
  `Kokkos::TeamPolicy<ExecSpace, LaunchBounds<MT,1>>(instance, nblocks, MT)`
  kernels with one element per team thread. The index decomposition
  (`BoxIndexer`) and the per-thread RNG state indexing are the same as
  before, so `ParallelForRNG` produces bit-identical random numbers.
- Why `TeamPolicy` and not `RangePolicy`/`MDRangePolicy`: a `ParallelFor`
  lambda may take a `Gpu::Handler`, and nvcc cannot tell on the host whether
  a `__device__` lambda takes one (`std::is_invocable` is false for every
  signature in the host pass). The decision has to be made in device code,
  where the team member is available. `TeamPolicy` and `RangePolicy`
  measured the same speed.
- `Gpu::Handler` wraps the Kokkos team member.
- New block-level launch API: `Gpu::team_launch<MT>(nblocks, shmem_bytes,
  f(team_member))` and `Gpu::handler_launch<MT>(nblocks, shmem_bytes,
  f(handler))`. The handler provides `blockIdx()`, `threadIdx()`,
  `blockDim()`, `gridDim()`, `globalIdx()`, `sharedMemory()` (team scratch)
  and `syncThreads()`.
- AMReX's internal block-level kernels use them: fused `MultiFab`
  `ParallelFor` (including strided and red-black), `TagParallelFor`, the
  FillBoundary copy and unpack kernels (`AMReX_FBI.H`,
  `AMReX_FBUnpackOnArrival.H`), `Gpu::fillAsync`, `ParReduce`,
  `BaseFabUtility`, `MultiFabUtil`, `TagBox`, `MLTensorOp`,
  `MLEBTensorOp_bc`, and the single-box BiCGStab solver
  (`AMReX_SingleBoxCGSolver.H`), which now also builds for SYCL.

### Reductions

- Each `ReduceOps::eval` is an asynchronous `Kokkos::parallel_reduce` with a
  custom reducer into a per-stream device accumulator. The reducer's
  `final()` combines the result with earlier `eval`s on the device, so
  several `eval`s still need no host synchronization. `ReduceData::value()`
  copies the accumulators to pinned memory and combines the streams.
- Register accumulation: the reduction runs over
  `nt = max(1, min(concurrency, n/4))` work items, each of which loops over a
  strided subset of the elements in registers. The policy is
  `RangePolicy<IndexType<Long>, LaunchBounds<AMREX_GPU_MAX_THREADS,1>>`.
- `Gpu::blockReduce*` and `Gpu::deviceReduce*(dest, value, handler)` use
  `team_reduce` (a full block does one atomic, a partial block per-thread
  atomics).
- `AnyOf` is a `ParallelFor` with a flag set by
  `Kokkos::atomic_load`/`atomic_store`.

### Atomics

- The device paths of `Gpu::Atomic::*` (`Add`, `Min`, `Max`, `LogicalOr`,
  `LogicalAnd`, `Exch`, `CAS`, `Multiply`, `Divide`, ...) use Kokkos'
  atomics (desul); the remaining compare-and-swap loops use
  `Kokkos::atomic_compare_exchange`. `AMReX_GpuAtomic.H` went from 757 to
  511 lines.
- The HIP gfx90a `unsafeAtomicAdd` overloads stay (faster floating-point add
  on MI200). The SYCL address-space template parameter stays for API
  compatibility.

### Memory

- The arenas allocate device, managed and pinned memory from Kokkos memory
  spaces (`CudaSpace`/`HIPSpace`/SYCL USM device space, `SharedSpace`,
  `SharedHostPinnedSpace`). On out-of-memory, AMReX still frees unused arena
  memory and retries.
- `Gpu::htod_memcpy*`, `dtoh_memcpy*` and `dtod_memcpy*` use
  `Kokkos::deep_copy` on the stream's instance.

## What stays native, and why

| Feature | Reason |
| --- | --- |
| Prefix sums (`Scan::PrefixSum`, `Gpu::exclusive_scan`, `inclusive_scan`) | `Kokkos::parallel_scan` is about 9x slower than AMReX's single-pass scan on CUDA. The best scan built from Kokkos team primitives was 1.19x slower (see attempts below). |
| RNG engines (curand/hiprand/oneMKL) | Kokkos' Random has no Poisson distribution, and a different generator would change every random sequence. Only the launch of `ParallelForRNG` moved to Kokkos. |
| `amrex::launch(nblocks, ...)` with a block count, legacy `AMREX_LAUNCH_*` macros, `single_task` | User lambdas use native `threadIdx.x`/`blockIdx.x`. Kokkos maps team threads to `threadIdx.y`, so porting these would break user code. |
| Warp-intrinsic and CUB/rocPRIM block algorithms (`DenseBins` warp-aggregated atomics, `ParticleUtil` block scans, CSR segmented sort) | Kokkos has no portable warp intrinsics or block-level sort and scan primitives with the same performance. |
| Raw CUDA/HIP block-reduce primitives in `AMReX_GpuReduce.H` | Still used by the kernels in the previous row and by OpenBC. |
| CPU loops (`ParallelFor` and the other launch functions in CPU builds) | `MDRangePolicy<Serial>` was 4.5x slower than AMReX's loop nests (the innermost loop does not vectorize). `RangePolicy<Serial>` dispatch costs 1-5% and takes an instance mutex per launch. |
| FFT (cuFFT/rocFFT/oneMKL) | Outside the portability layer; kokkos-fft could be evaluated separately. |
| OpenBC `compute_potential` | Still aborts on SYCL, as in `development`. It could use `handler_launch`, but only an FFT-enabled test covers it. |

## Performance

### Setup

- **Hardware:** NVIDIA RTX A2000 8GB Laptop GPU (Ampere, sm_86, driver
  620.70.06); Intel Core i9-12900H, runs pinned to one P-core (`taskset -c
  10`).
- **Software:** CUDA 12.9.86, GCC 13.3.0, CMake 3.30.4 (micromamba
  environments `warpx-gpu-mpich-dev` and `warpx-cpu-mpich-dev`).
- **Kokkos 5.3.0:** Release, static. CUDA build: `Kokkos_ENABLE_SERIAL=ON`,
  `Kokkos_ENABLE_CUDA=ON`, `Kokkos_ARCH_AMPERE86=ON`,
  `Kokkos_ENABLE_COMPILE_AS_CMAKE_LANGUAGE=ON`, no RDC. CPU build:
  `Kokkos_ENABLE_SERIAL=ON`, `Kokkos_ENABLE_OPENMP=ON`.
- **AMReX:** Release, 3D, double precision, `AMReX_MPI=OFF`,
  `AMReX_GPU_RDC=OFF`, for both the baseline (`development` at `ff7868ce94`)
  and `topic-kokkos`. CPU builds use `AMReX_OMP=ON` and run with
  `OMP_NUM_THREADS=1`.
- **Benchmark:** a standalone program (source in the appendix) built
  against each installed AMReX. Each case is warmed up once and then timed
  over `nrep` repetitions with a stream synchronization at both ends. Each
  build ran as 5 separate processes, alternating which build ran first in a
  round. GPU: `n_cell=256` (256³ cells), `max_grid_size=64` (64 boxes of
  64³; 4096 boxes of 16³ in `small_launches_16cubed`), `nrep=20`. CPU:
  `n_cell=128`, `max_grid_size=32` (64 boxes of 32³; 512 boxes of 16³),
  `nrep=20`, `OMP_NUM_THREADS=1`. The flat-array cases use the same sizes on
  CPU and GPU.
- **Ratio** = Kokkos time / `development` time, so lower is better. "Best"
  compares the fastest of the 5 runs; "median" compares the medians.

### GPU results

| Case | Measures | `development` best (ms) | Kokkos best (ms) | Ratio, best | Ratio, median |
| --- | --- | ---: | ---: | ---: | ---: |
| `stencil_mfiter` | 7-point stencil, `MFIter` loop with one `ParallelFor(Box)` per box | 1.811 | 1.837 | 1.014 | 1.016 |
| `stencil_fused` | same stencil, one `ParallelFor(MultiFab)` for all boxes | 2.111 | 2.098 | 0.994 | 0.991 |
| `small_launches_16cubed` | `MFIter` loop over 16³ boxes, one `ParallelFor(Box, 2 comps)` each | 4.917 | 6.016 | 1.224 | 1.235 |
| `mf_sum` | `MultiFab::sum` | 0.902 | 0.856 | 0.949 | 0.948 |
| `mf_norm0` | `MultiFab::norm0` | 0.906 | 0.861 | 0.950 | 0.951 |
| `mf_norm1` | `MultiFab::norm1` | 0.909 | 0.868 | 0.955 | 0.955 |
| `mf_min_max` | `MultiFab::min` + `MultiFab::max` | 1.820 | 1.751 | 0.962 | 0.959 |
| `reduceops_sum_min_max_mf` | `ReduceOps<Sum,Min,Max>` over a `MultiFab` | 1.116 | 1.261 | 1.130 | 1.111 |
| `reduce_sum_16M` | `Reduce::Sum`, 16M doubles | 0.792 | 0.800 | 1.010 | 1.010 |
| `reduceops_sum_min_max_16M` | `ReduceOps<Sum,Min,Max>`, 16M doubles | 1.009 | 1.029 | 1.020 | 1.006 |
| `exclusive_scan_16M` | `Gpu::exclusive_scan`, 16M ints | 0.819 | 0.819 | 1.001 | 0.999 |
| `prefixsum_retsum_16M` | `Scan::PrefixSum` with `retSum`, 16M ints | 0.832 | 0.832 | 1.000 | 0.999 |
| `atomic_add_deposit_16M` | 16M `Gpu::Atomic::AddNoRet` scattered over a 128³ grid | 18.138 | 18.024 | 0.994 | 0.994 |
| `atomic_max_16M` | 16M `Gpu::Atomic::Max` onto 64 addresses | 1.604 | 0.778 | 0.485 | 0.483 |
| `fillboundary` | `MultiFab::FillBoundary`, periodic, 1 ghost cell | 0.376 | 0.379 | 1.008 | 1.009 |
| `fillboundary_nonperiodic` | `MultiFab::FillBoundary`, non-periodic | 0.313 | 0.318 | 1.014 | 1.009 |
| `rng_4M` | `ParallelForRNG`, 4M uniform random numbers | 1.926 | 1.054 | 0.547 | 0.537 |

Both builds produce bit-identical random numbers in `rng_4M` (same sum,
mean and variance to 17 digits).

### CPU results

The CPU build with Kokkos uses `AMReX_KOKKOS=ON` (Kokkos with Serial and
OpenMP). AMReX's CPU code paths for loops, reductions, scans, atomics and
the arenas are the same as in `development`; only Kokkos' initialization is
added. The differences below (median ratios from 0.990 to 1.018) are the
run-to-run noise of this laptop CPU. The default CPU build
(`AMReX_KOKKOS=OFF`) does not use Kokkos at all.

| Case | `development` best (ms) | Kokkos best (ms) | Ratio, best | Ratio, median |
| --- | ---: | ---: | ---: | ---: |
| `stencil_mfiter` | 1.552 | 1.520 | 0.979 | 0.996 |
| `stencil_fused` | 1.554 | 1.569 | 1.010 | 1.000 |
| `small_launches_16cubed` | 1.518 | 1.498 | 0.987 | 0.990 |
| `mf_sum` | 0.525 | 0.526 | 1.001 | 1.015 |
| `mf_norm0` | 1.331 | 1.347 | 1.012 | 1.006 |
| `mf_norm1` | 0.510 | 0.512 | 1.005 | 0.996 |
| `mf_min_max` | 1.019 | 0.991 | 0.972 | 0.999 |
| `reduceops_sum_min_max_mf` | 0.535 | 0.541 | 1.011 | 1.018 |
| `reduce_sum_16M` | 6.062 | 6.061 | 1.000 | 1.002 |
| `reduceops_sum_min_max_16M` | 6.732 | 6.724 | 0.999 | 1.004 |
| `exclusive_scan_16M` | 6.253 | 6.280 | 1.004 | 0.997 |
| `prefixsum_retsum_16M` | 6.207 | 6.218 | 1.002 | 1.000 |
| `atomic_add_deposit_16M` | 58.710 | 58.539 | 0.997 | 1.015 |
| `atomic_max_16M` | 10.422 | 10.426 | 1.000 | 1.003 |
| `fillboundary` | 0.768 | 0.767 | 0.999 | 0.991 |
| `fillboundary_nonperiodic` | 0.562 | 0.556 | 0.990 | 1.000 |
| `rng_4M` | 32.364 | 32.195 | 0.995 | 0.993 |

### The slower GPU cases

**Many tiny kernels (1.22x best, 1.24x median).** 4096 boxes of 16³ cells,
one kernel each: the GPU work per kernel is shorter than the host's enqueue
time, so the host launch cost decides. Measured host time per enqueue on an
idle queue:

| Launch | µs per launch |
| --- | --- |
| Raw CUDA launch | 1.21 |
| Raw CUDA launch with a 256-byte argument, `cudaSetDevice` and 2 KB dynamic shared memory (mimicking a Kokkos launch) | 1.29 |
| `Kokkos::parallel_for` with `TeamPolicy` | 2.01–2.06 |
| Calling `Kokkos::Impl::ParallelFor::execute` directly (skipping dispatch and tools hooks) | 1.95 |
| Patched Kokkos without the `team_size_max` check | 1.92 |
| Patched Kokkos without the static launch mutex | 2.01 |
| Patched Kokkos without the per-launch `cudaSetDevice` | 1.94 |
| Patched Kokkos with all three removed | 1.83 |

In Kokkos 5.3.0, every `TeamPolicy` launch also calls `CudaSpace::allocate(0)`
(`resize_team_scratch_space` keeps a size of zero and reallocates); Kokkos'
`develop` branch (5.3.99) already skips that allocation for all GPU
backends, but its cost was not isolated in these measurements. About
0.5 µs remain unexplained: a CPU profiler was not available
(`perf_event_paranoid=4`). `RangePolicy` and `TeamPolicy` cost the same,
and `Kokkos::Experimental::WorkItemProperty::HintLightWeight` did not help.
For applications, `ParallelFor` over a whole `MultiFab` (`MultiFab::arrays`)
launches one kernel for all boxes and avoids the overhead (`stencil_fused`
is at parity).

**Fused three-value `ReduceOps` over a `MultiFab` (1.13x best, 1.11x
median).** Kokkos keeps the per-thread values of a multi-word reduction in
shared memory (`UseShflReduction=false`), while AMReX's native kernel kept
them in registers. Attempts:

| Variant | Ratio |
| --- | --- |
| `RangePolicy`, one element per work item | 1.30 |
| Register accumulation (`nt = min(concurrency, n/4)` work items) | 1.12 |
| Plus `LaunchBounds<256,1>` (kept) | 1.10–1.13 across benchmark sessions |
| 4x or 8x more work items | worse |
| `LaunchBounds` with 2 or 4 minimum blocks per SM | no gain |
| `TeamPolicy` kernel shaped like the native one | 1.12 |

Single-value reductions (`MultiFab::sum`, `norm0`, `norm1`, `min`/`max`)
are faster, and the same three-value reduction over a flat 16M array is at
parity (1.006x median).

**Small differences (1–2%).** `stencil_mfiter` (64 launches per iteration,
spread over AMReX's GPU streams) is 1.4–1.6% slower, and FillBoundary is
about 1% slower. Exposing part of the per-launch host overhead, e.g. at the
start and end of the `MFIter` loop, would fit these numbers, but they were
not root-caused. The same stencil as one fused `ParallelFor(MultiFab)`
(`stencil_fused`) is at parity (0.99x).

### Prefix sum attempts (kept native)

| Variant (16M `int`, exclusive) | Ratio to AMReX's scan |
| --- | --- |
| `Kokkos::parallel_scan` | about 9x |
| `Kokkos::parallel_scan` with `LaunchBounds<1024>` | no gain |
| Single-pass `TeamPolicy` scan with decoupled look-back, a `team_scan` per element | about 7x |
| Same, with a tile in team scratch and one `team_scan` per tile | 1.45x |
| Same, with an odd number of elements per thread | 1.19x (1.10x at 1M) |
| Windowed parallel look-back | 2x |
| Larger tiles | 1.19–1.27x |

### Notes on the faster GPU cases

- `atomic_max_16M` (2x faster): Kokkos' floating-point `atomic_max` skips
  the compare-and-swap when the stored value is already larger, while
  AMReX's old loop always swapped. In a separate microbenchmark the early
  exit was 10x faster on one hot address, but 10–34% slower when many
  addresses are updated and every update changes the value. Which pattern
  dominates depends on the application.
- `rng_4M` (1.8x faster): the kernel does the same work as before (same
  grid-stride loop, same RNG states, identical output); the cause of the
  speed-up was not investigated.
- `mf_sum`, `mf_norm*`, `mf_min_max` (4–5% faster): single-value reductions
  with register accumulation.

## Testing

All runs were on the machine described above.

| Configuration | Command | Result |
| --- | --- | --- |
| CUDA, installed Kokkos 5.3.0 (no RDC), `AMReX_GPU_RDC=OFF`, `AMReX_TEST_TYPE=All`, `AMReX_PARTICLES=ON`, `AMReX_EB=ON`, `AMReX_LINEAR_SOLVERS=ON`, MPI off | `ctest --test-dir build --output-on-failure -j1` | 136/136 passed |
| CPU + OpenMP, Kokkos downloaded and built by AMReX (`AMReX_KOKKOS=ON`, `AMReX_kokkos_internal=ON`), `AMReX_TEST_TYPE=All`, same components | `ctest --test-dir build --output-on-failure -j3` | 138/138 passed |
| CUDA default configuration (`AMReX_GPU_RDC=ON`, Kokkos built by AMReX from a local source tree with RDC), `AMReX_TEST_TYPE=Small` | `ctest --test-dir build --output-on-failure -j1` | 9/9 passed |
| GNU make, `Tests/GPU/AnyOf` | `make -j6 USE_CUDA=TRUE USE_GPU_RDC=FALSE CUDA_ARCH=86 USE_MPI=FALSE KOKKOS_HOME=<kokkos>` | builds with `--Werror cross-execution-space-call`, runs correctly |

The CUDA tests run one at a time because each AMReX process reserves 3/4 of
the GPU memory by default, so parallel `ctest` runs out of memory on an 8 GB
GPU. This is the same with `development`.

## Limitations and risks

- **HIP and SYCL are unverified.** The HIP and SYCL branches were written
  against the Kokkos API, but never compiled or run.
- **MPI was off in every test.** Code paths that only run with several
  ranks, e.g. unpacking received FillBoundary/ParallelCopy buffers
  (`AMReX_FBUnpackOnArrival.H`), were compiled but not exercised.
- **One GPU model.** All numbers come from a laptop RTX A2000. The launch
  overhead depends on the host CPU, and the atomic and reduction results
  may differ on A100/H100/MI250X/PVC.
- **One GPU architecture per build.** Kokkos supports one architecture per
  build, so AMReX can no longer build fat binaries for several
  `CMAKE_CUDA_ARCHITECTURES`.
- **New required dependency for GPU builds.** Without an installed Kokkos,
  configuring needs network access (or `AMReX_kokkos_src`). Kokkos and
  AMReX must agree on the RDC mode, and a CUDA Kokkos needs
  `Kokkos_ENABLE_COMPILE_AS_CMAKE_LANGUAGE=ON`; both are checked with clear
  errors. GNU make defaults to `USE_GPU_RDC=TRUE` and therefore needs an RDC
  Kokkos.
- **Host launch overhead** of about 0.8 µs per kernel, which matters for
  `MFIter` loops over many very small boxes.
- **Floating-point atomic min/max** speed depends on the update pattern
  (see above).
- **Not all native launch code is gone:** raw `amrex::launch` with a block
  count, the legacy macros, the warp-intrinsic kernels, the scans and the RNG
  engines keep their CUDA/HIP/SYCL code (see the table above).

## Suggestions for Kokkos upstream

1. `resize_team_scratch_space`: skip the allocation when the size is 0.
   Already done in Kokkos' `develop` branch (after 5.3.0) for CUDA, HIP and
   SYCL.
2. The RDC mismatch `#error` in desul (vendored in Kokkos as
   `tpls/desul`) reads "Relocatable device code mode incompatible with desul
   atomics configuration". It could say which side uses relocatable device
   code and name the fix (`Kokkos_ENABLE_CUDA_RELOCATABLE_DEVICE_CODE` /
   `Kokkos_ENABLE_HIP_RELOCATABLE_DEVICE_CODE`, or the compiler's RDC flag).
3. Per-launch host work in the CUDA `TeamPolicy` path that is unchanged in
   `develop`: the `team_size_max` re-check (0.09 µs) and `cudaSetDevice`
   (0.07 µs). Caching or skipping them needs care: the shared memory a
   functor requests can depend on the functor instance, and the per-launch
   `cudaSetDevice` supports instances on several devices.
4. CUDA `parallel_scan`: a single-pass decoupled look-back scan (CUB-like)
   would close the 9x gap.
5. `RangePolicy` reductions of multi-word values: accumulate the per-thread
   value in registers (`UseShflReduction=false` keeps it in shared memory).

## Possible next steps

- Compile and test HIP (MI250X/MI300) and SYCL (PVC), and run the MPI tests.
- Repeat the benchmark on data-center GPUs and with a WarpX/ImpactX
  workload.
- Port the OpenBC `compute_potential` kernel to `handler_launch` (enables
  SYCL), with an FFT-enabled test.
- Report the upstream issues above, and revisit the prefix sum and the
  launch overhead once Kokkos improves.

## Appendix: benchmark source

<details>
<summary><code>bench.cpp</code> and <code>CMakeLists.txt</code></summary>

`CMakeLists.txt`:

```cmake
cmake_minimum_required(VERSION 3.25)
project(amrex_portability_bench LANGUAGES C CXX)
find_package(AMReX REQUIRED)
if (AMReX_GPU_BACKEND STREQUAL "CUDA")
    enable_language(CUDA)
    set_source_files_properties(bench.cpp PROPERTIES LANGUAGE CUDA)
endif ()
add_executable(bench bench.cpp)
target_link_libraries(bench PRIVATE AMReX::amrex)
if (AMReX_GPU_BACKEND STREQUAL "CUDA")
    set_target_properties(bench PROPERTIES CUDA_ARCHITECTURES 86)
endif ()
```

`bench.cpp`:

```cpp
// AMReX portability-layer benchmark: run against a baseline build and a Kokkos build.
#include <AMReX.H>
#include <AMReX_MultiFab.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Print.H>
#include <AMReX_Reduce.H>
#include <AMReX_Scan.H>
#include <AMReX_Random.H>
#include <AMReX_GpuContainers.H>

#include <chrono>
#include <iomanip>
#include <functional>
#include <string>
#include <vector>

using namespace amrex;

namespace {
double time_it (int nrep, std::function<void()> const& f)
{
    f(); // warm up
    Gpu::streamSynchronize();
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < nrep; ++i) { f(); }
    Gpu::streamSynchronize();
    auto t1 = std::chrono::steady_clock::now();
    return std::chrono::duration<double,std::milli>(t1-t0).count() / nrep;
}

void report (std::string const& name, double ms) {
    amrex::Print() << "BENCH " << name << " " << ms << " ms\n";
}
}

int main (int argc, char* argv[])
{
    amrex::Initialize(argc, argv);
    {
        int n_cell = 256;
        int max_grid_size = 64;
        int nrep = 20;
        {
            ParmParse pp;
            pp.query("n_cell", n_cell);
            pp.query("max_grid_size", max_grid_size);
            pp.query("nrep", nrep);
        }
        Box domain(IntVect(0), IntVect(n_cell-1));
        BoxArray ba(domain);
        ba.maxSize(max_grid_size);
        DistributionMapping dm(ba);
        MultiFab a(ba, dm, 1, 1);
        MultiFab b(ba, dm, 1, 1);
        a.setVal(1.0);
        b.setVal(0.0);
        auto const& aa = a.arrays();
        ParallelFor(a, IntVect(1), [=] AMREX_GPU_DEVICE (int bi, int i, int j, int k) {
            aa[bi](i,j,k) = 1.0 + 0.001*(i%17) + 0.002*(j%13) - 0.003*(k%11);
        });

        // 1. 7-point stencil with MFIter + ParallelFor per box
        report("stencil_mfiter", time_it(nrep, [&] {
            for (MFIter mfi(b); mfi.isValid(); ++mfi) {
                Box const& bx = mfi.validbox();
                auto const& ar = a.const_array(mfi);
                auto const& br = b.array(mfi);
                ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) {
                    br(i,j,k) = ar(i,j,k) - (1./6.)*(ar(i-1,j,k)+ar(i+1,j,k)+ar(i,j-1,k)
                                                    +ar(i,j+1,k)+ar(i,j,k-1)+ar(i,j,k+1));
                });
            }
        }));

        // 2. fused MultiFab ParallelFor
        report("stencil_fused", time_it(nrep, [&] {
            auto const& ba_ = a.const_arrays();
            auto const& bb_ = b.arrays();
            ParallelFor(b, [=] AMREX_GPU_DEVICE (int bi, int i, int j, int k) {
                auto const& ar = ba_[bi];
                bb_[bi](i,j,k) = ar(i,j,k) - (1./6.)*(ar(i-1,j,k)+ar(i+1,j,k)+ar(i,j-1,k)
                                                      +ar(i,j+1,k)+ar(i,j,k-1)+ar(i,j,k+1));
            });
            Gpu::streamSynchronize();
        }));

        // 3. many small launches (16^3 boxes, ncomp loop)
        {
            BoxArray bas(domain);
            bas.maxSize(16);
            DistributionMapping dms(bas);
            MultiFab s(bas, dms, 2, 0);
            s.setVal(0.0);
            report("small_launches_16cubed", time_it(nrep, [&] {
                for (MFIter mfi(s); mfi.isValid(); ++mfi) {
                    auto const& sr = s.array(mfi);
                    ParallelFor(mfi.validbox(), 2, [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) {
                        sr(i,j,k,n) += 1.0;
                    });
                }
            }));
            amrex::Print() << "INFO small boxes: " << bas.size() << "\n";
        }

        // 4. reductions
        report("mf_sum", time_it(nrep, [&] { volatile Real r = a.sum(0); amrex::ignore_unused(r); }));
        report("mf_norm0", time_it(nrep, [&] { volatile Real r = a.norm0(0); amrex::ignore_unused(r); }));
        report("mf_norm1", time_it(nrep, [&] { volatile Real r = a.norm1(0); amrex::ignore_unused(r); }));
        report("mf_min_max", time_it(nrep, [&] { volatile Real r = a.min(0) + a.max(0); amrex::ignore_unused(r); }));
        report("reduceops_sum_min_max_mf", time_it(nrep, [&] {
            ReduceOps<ReduceOpSum, ReduceOpMin, ReduceOpMax> reduce_op;
            ReduceData<Real, Real, Real> reduce_data(reduce_op);
            using ReduceTuple = typename decltype(reduce_data)::Type;
            auto const& ma = a.const_arrays();
            reduce_op.eval(a, IntVect(0), reduce_data,
                [=] AMREX_GPU_DEVICE (int bi, int i, int j, int k) -> ReduceTuple {
                    Real x = ma[bi](i,j,k);
                    return {x, x, x};
                });
            auto hv = reduce_data.value(reduce_op);
            volatile Real r = amrex::get<0>(hv) + amrex::get<1>(hv) + amrex::get<2>(hv);
            amrex::ignore_unused(r);
        }));
        {
            Long n = Long(1) << 24;
            Gpu::DeviceVector<Real> v(n);
            Real* pv = v.data();
            ParallelFor(n, [=] AMREX_GPU_DEVICE (Long i) { pv[i] = Real(i%7)*0.5; });
            report("reduce_sum_16M", time_it(nrep, [&] {
                volatile Real r = Reduce::Sum(n, pv); amrex::ignore_unused(r); }));
            report("reduceops_sum_min_max_16M", time_it(nrep, [&] {
                ReduceOps<ReduceOpSum, ReduceOpMin, ReduceOpMax> reduce_op;
                ReduceData<Real, Real, Real> reduce_data(reduce_op);
                using ReduceTuple = typename decltype(reduce_data)::Type;
                reduce_op.eval(n, reduce_data, [=] AMREX_GPU_DEVICE (Long i) -> ReduceTuple {
                    return {pv[i], pv[i], pv[i]};
                });
                auto hv = reduce_data.value(reduce_op);
                volatile Real r = amrex::get<0>(hv); amrex::ignore_unused(r);
            }));

            // 5. scans
            Gpu::DeviceVector<int> iv(n), ov(n);
            int* piv = iv.data();
            int* pov = ov.data();
            ParallelFor(n, [=] AMREX_GPU_DEVICE (Long i) { piv[i] = int(i%3); });
            report("exclusive_scan_16M", time_it(nrep, [&] {
                Gpu::exclusive_scan(iv.begin(), iv.end(), ov.begin()); }));
            report("prefixsum_retsum_16M", time_it(nrep, [&] {
                volatile int r = Scan::PrefixSum<int>(int(n),
                    [=] AMREX_GPU_DEVICE (int i) -> int { return piv[i]; },
                    [=] AMREX_GPU_DEVICE (int i, int const& x) { pov[i] = x; },
                    Scan::Type::exclusive, Scan::retSum);
                amrex::ignore_unused(r); }));

            // 6. atomics: deposition of 16M "particles" onto a 128^3 grid
            int ng = 128;
            Gpu::DeviceVector<Real> grid(ng*ng*ng, 0.0);
            Real* pg = grid.data();
            report("atomic_add_deposit_16M", time_it(nrep, [&] {
                ParallelFor(n, [=] AMREX_GPU_DEVICE (Long i) {
                    auto c = (i * 2654435761L) % (Long(ng)*ng*ng);
                    Gpu::Atomic::AddNoRet(pg + c, Real(1.0));
                });
            }));
            report("atomic_max_16M", time_it(nrep, [&] {
                ParallelFor(n, [=] AMREX_GPU_DEVICE (Long i) {
                    Gpu::Atomic::Max(pg + (i%64), pv[i]);
                });
            }));
        }

        // 7. FillBoundary
        report("fillboundary", time_it(nrep, [&] { a.FillBoundary(Periodicity(IntVect(n_cell))); }));
        report("fillboundary_nonperiodic", time_it(nrep, [&] { a.FillBoundary(); }));

        // 8. RNG
        {
            Long n = Long(1) << 22;
            Gpu::DeviceVector<Real> v(n);
            Real* pv = v.data();
            report("rng_4M", time_it(nrep, [&] {
                ParallelForRNG(n, [=] AMREX_GPU_DEVICE (Long i, RandomEngine const& engine) {
                    pv[i] = Random(engine);
                });
            }));
            {
                Real sum = Reduce::Sum(n, pv);
                Real sum2 = Reduce::Sum<Real>(n, [=] AMREX_GPU_DEVICE (Long i) { return pv[i]*pv[i]; });
                Real mean = sum/Real(n);
                amrex::Print() << std::setprecision(17) << "INFO rng_check sum " << sum
                               << " mean " << mean << " var " << sum2/Real(n) - mean*mean << "\n";
            }
        }
    }
    amrex::Finalize();
}
```

</details>
