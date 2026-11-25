# NCCL Experiment Summary

Investigation of 2026-11-25: can AMReX benefit from NVIDIA NCCL for guard-cell and particle
exchange between GPUs, next to or instead of GPU-aware MPI?

> The measurements below come from an early-access system (cech, GB200) that is under NDA.
> Check before sharing this file or its numbers outside the group.

## Verdict

**Partly.** NCCL does not move bytes faster than GPU-aware MPI (HPC-X) on GB200. What it buys is
fewer host round trips per exchange, and that only pays off for small and medium guard exchanges.

- Small-halo `FillBoundary`: 1.4–2.1x faster on one node, 5–33% faster across two nodes.
- Large-halo `FillBoundary`: slower with NCCL (10–55% on one node, 1.8x across two nodes).
- Particle `Redistribute`: nothing to gain from swapping the transport.

## Setup

| | |
|---|---|
| Nodes | 4x NVIDIA GB200 (189 GB each) and 144 Arm cores in two sockets per node |
| GPU links | NVLink between the 4 GPUs of a node; no multi-node NVLink fabric reported |
| Network | InfiniBand, 2x 400 Gb/s per node (one NIC per GPU pair) |
| Software | NVHPC 26.9, CUDA 13.3, NCCL 2.31.2 (bundled; newest release is 2.32.7), HPC-X 2.50 (Open MPI 5.0.10, UCX 1.21), GCC 14.4.1 |
| AMReX | `development` at 492c757e9b plus the prototype of this branch; CUDA, MPI, double precision, 3D |
| Runs | one MPI rank per GPU, 8 cores per rank, ranks bound to the GPU's socket |

All MPI runs use `UCX_MEMTYPE_CACHE=n` (see "Other findings").

## FillBoundary in AMReX

Microseconds per `FillBoundary` call of one `MultiFab` on a periodic domain. Every completed run
passed a check of all ghost cells (0 wrong cells).

| Case | 1 node, MPI | 1 node, `ncclSend`/`ncclRecv` | 1 node, NVLink peer writes | 2 nodes, MPI | 2 nodes, `ncclSend`/`ncclRecv` |
|---|---|---|---|---|---|
| 128³, 64³ boxes, 1 comp, 1 ghost | 64 | 33 | 30 | 90 | 68 |
| 256³, 128³ boxes, 1 comp, 1 ghost | 65 | 40 | 37 | 97 | 91 |
| 256³, 512 boxes of 32³, 1 comp, 1 ghost | 68 | 50 | 46 | 105 | 86 |
| 512³, 256³ boxes, 3 comps, 2 ghosts | 171 | 190 | 192 | 340 | 624 |
| 512³ nodal, 128³ boxes, 6 comps, 4 ghosts | 549 | 733 | 856 | 1600 | crashed |

One node is 4 ranks, two nodes are 8 ranks (4 per node).

- **Where the gain comes from.** The MPI path is pack, stream synchronization, post and wait,
  unpack, stream synchronization. Both NCCL paths queue everything on the GPU stream and
  synchronize once. In the peer-write mode the pack kernel writes straight into the neighbor GPU,
  and flags in peer memory replace the handshake.
- **Why large halos lose.** They are bandwidth-bound, and UCX's bulk GPU-to-GPU copy together
  with AMReX's unpack-on-arrival is already better. A real implementation needs a switch by
  message size.
- **Peer writes stop at the node here.** NCCL's load/store team is the 4 GPUs of a node on this
  machine, so that mode was not run across nodes.
- **The crash** is a segmentation fault inside HPC-X's bundled NCCL network plugin
  (`ncclIbMultiSend` in `nccl_spectrum-x_plugin/libnccl-net.so`), on NCCL's progress thread. It
  was not retried with NCCL's internal InfiniBand transport.
- **Baseline profile.** In the 256³ MPI case (70 µs in the profiled run), about 16 µs wait for
  receives, 13 µs pack, 10 µs post sends, 10 µs final synchronization, 4.5 µs unpack and 4 µs
  post receives.
  For the large nodal case, waiting for receives is 360 of 550 µs.

Runs are single measurements of 300–1000 calls. Repeats differed by up to about 10%. The
prototype is untuned.

## Transport micro-benchmark

Every rank exchanges one message with each peer, as a `FillBoundary` does: pack kernel, transfer,
unpack kernel. Microseconds per exchange, maximum over ranks, best of three repetitions.

### One node, 4 ranks, 3 NVLink peers per rank

| Chain | 4 KiB | 256 KiB | 2 MiB | 16 MiB |
|---|---|---|---|---|
| pack and unpack only, no transfer | 12 | 13 | 14 | 47 |
| MPI | 53 | 55 | 69 | 162 |
| `ncclSend`/`ncclRecv` | 21 | 29 | 44 | 173 |
| `ncclSend`/`ncclRecv` on symmetric windows | 21 | 30 | 40 | 151 |
| `ncclPutSignal`/`ncclWaitSignal` (host one-sided) | 40 | 48 | 77 | 290 |
| kernel-issued network puts (GIN), all peers through the NIC | 36 | 62 | 284 | 2215 |
| peer writes, completion by 1-byte MPI messages | 34 | 37 | 54 | 178 |
| peer writes, ready and done flags on the device | 20 | 23 | 41 | 166 |
| peer writes straight into the destination array | 16 | 18 | 36 | 134 |

Transfer alone, without pack and unpack: MPI 30, 32, 44, 103 µs and `ncclSend`/`ncclRecv`
18, 24, 39, 137 µs for the same four sizes. Without `UCX_MEMTYPE_CACHE=n`, MPI is about 7 µs
faster per small exchange (24 µs transfer, 45 µs chain at 4 KiB), where it does not crash.

### Two nodes, 2 ranks, 1 InfiniBand peer per rank

| Chain | 4 KiB | 256 KiB | 2 MiB | 16 MiB |
|---|---|---|---|---|
| MPI | 31 | 68 | 118 | 577 |
| `ncclSend`/`ncclRecv` | 30 | 46 | 93 | 411 |
| `ncclSend`/`ncclRecv` on symmetric windows | 29 | 38 | 77 | 387 |
| `ncclPutSignal`/`ncclWaitSignal` | 31 | 34 | 74 | 386 |
| kernel-issued network puts (GIN) | 28 | 35 | 75 | 394 |

Transfer alone at 4 KiB: MPI 10 µs (eager), NCCL 25–27 µs. So across InfiniBand NCCL is
1.3–2x faster from 256 KiB to 16 MiB, and equal on the full chain for tiny messages.

### Two nodes, 8 ranks, 7 peers per rank (3 NVLink, 4 InfiniBand)

| Chain | 4 KiB | 256 KiB | 2 MiB | 16 MiB |
|---|---|---|---|---|
| MPI | 76 | 135 | 695 | 3837 |
| `ncclSend`/`ncclRecv` | 51 | 122 | 737 | 3401 |
| kernel-issued network puts (GIN) | 53 | 120 | 654 | 5670 |

In this pattern `ncclSend`/`ncclRecv` showed erratic outliers at 256 KiB (340–810 µs in the
transfer-only and symmetric-window variants), so no conclusion is drawn from it beyond the
4 KiB column.

### Start-up cost of NCCL

Communicator creation took 1.3–2.4 s for 2–8 ranks, registering four windows 0.03–0.7 s, and
the device communicator for GIN 0.1–0.5 s. NCCL's own notes say host one-sided contexts connect
every rank to every rank, which scales as ranks squared.

## Particles and collectives

- **Redistribute.** About 7–8 ms per call for 8.4 million particles per GPU on one node:
  size handshake 2.1 ms, waiting for the particle messages 2.3 ms, partitioning 1.4 ms, pack
  and post 0.9 ms. The handshake and the waits absorb waiting for slower peers. NCCL gains tens
  of microseconds per exchange and cannot receive messages of unknown size, so the handshake
  stays in MPI. A one-sided redesign that removes the handshake is the only angle; it was not
  prototyped.
- **Collectives.** Not measured. AMReX's reductions act on host scalars after a GPU reduction,
  so NCCL's collectives on GPU buffers have almost nothing to act on.

## Other findings

- **GPU-aware MPI crashes by default on this stack.** UCX treats AMReX's small GPU
  communication buffers as host memory: one case segfaults in a CPU `memcpy` on a device pointer,
  another fails with an InfiniBand "local protection error". Large-halo cases run. Setting
  `UCX_MEMTYPE_CACHE=n` fixes all cases and costs about 7 µs per small exchange. Worth reporting
  to the centre.
- **UCX tuning does not help MPI.** `UCX_RNDV_THRESH`, `UCX_RNDV_SCHEME` and
  `UCX_PROTO_ENABLE=n` were all equal to or worse than the defaults.
- **MPI on CUDA VMM memory** (`ncclMemAlloc`, needed for NCCL windows) is up to 2x slower than on
  `cudaMalloc` memory for 256 KiB and larger messages on one node.
- **NCCL constraints for AMReX.**
  - One MPI rank per GPU.
  - Send and receive must match in size and type; no tags, no wildcards, no probe.
  - Peer windows need CUDA VMM memory, which AMReX arenas do not provide.
  - Window registration and device communicator creation are collective calls.
  - NCCL warns against concurrent MPI and NCCL transfers between the same GPUs.
- **Prior art.** Gkeyll and code_saturne ship `ncclSend`/`ncclRecv` ghost exchange. NVIDIA's own
  halo-exchange work (GROMACS) uses NVSHMEM instead, which was not tested here. No published
  comparison of NCCL point-to-point against GPU-aware MPI was found.

## Recommendation

Do not build a general NCCL backend. The worthwhile piece is an opt-in fast path for
latency-bound `FillBoundary` and `SumBoundary` (multigrid coarse levels, small boxes) with a
size threshold that falls back to MPI. Compare against NVSHMEM before committing to NCCL.

## The prototype on this branch

Experimental and opt-in. Configure with `-DAMReX_NCCL=ON -DNCCL_ROOT=<nccl prefix>` (needs CUDA
and MPI), then choose the path at run time:

| Input | Meaning |
|---|---|
| `amrex.nccl.fb_mode=0` | MPI (default) |
| `amrex.nccl.fb_mode=1` | pack, `ncclSend`/`ncclRecv`, unpack on the current GPU stream |
| `amrex.nccl.fb_mode=2` | pack kernels write into the receiving GPU over NVLink; needs all ranks in one NVLink load/store domain |
| `amrex.nccl.window_mb` | size of each rank's receive window for mode 2 (default 256) |
| `amrex.nccl.fence=0` | drop the store fence in the peer-write kernel, for experiments only |

Files: `Src/Base/AMReX_NCCL.{H,cpp}` (communicator, windows, flags), `Src/Base/AMReX_FBNccl.H`
(the two paths), hooks in `AMReX_FabArrayCommI.H`, `AMReX_FabArray.H` and `AMReX.cpp`.

Limits:

- Only `FillBoundary` and `SumBoundary` on GPU, with the same conditions as unpack-on-arrival
  (not deterministic, no CUDA graph). `ParallelCopy` and particles still use MPI.
- No switch by message size, so large halos get slower.
- Mode 2 supports one `FillBoundary` in flight per rank and at most 32 neighbor ranks.
- The NCCL dependency is not exported to installed CMake configs. No documentation, no tests.

## Open items

- Retry the crashed two-node case with NCCL's internal InfiniBand transport.
- Measure on a system whose NVLink domain spans nodes, where peer writes could reach more GPUs.
- Hybrid of peer writes on the node and kernel-issued network puts off the node.
- NVSHMEM as an alternative to NCCL.
- NCCL 2.32, which adds peer pointers for any load/store-reachable peer and explicit peer lists
  for kernel-issued network puts.

## Where the runs are

On cech in `~/2026-11-25_nccl_dbdf7ada`: the micro-benchmark `bench/bench_p2p.cu`, the
`FillBoundary` benchmark `amrex/Tests/FillBoundaryBench`, the builds `build-base` and
`build-nccl`, the step scripts and all logs (`bench/*.log`, `runs*/`).
