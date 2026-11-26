# NVSHMEM Experiment Summary

Investigation of 2026-11-25/26: can AMReX benefit from NVIDIA NVSHMEM for guard-cell and particle
exchange between GPUs, next to or instead of GPU-aware MPI? Follow-up to the NCCL study on the
branch `topic-nccl` (`NCCL_Experiment_Summary.md`), on the same machine and with the same test
cases.

> The measurements below come from an early-access system (cech, GB200) that is under NDA.
> Check before sharing this file or its numbers outside the group.

## Verdict

**Partly, and for the same reason as NCCL.** NVSHMEM does not move bytes faster than GPU-aware
MPI (HPC-X) on GB200. It removes host round trips per exchange, and that only pays off for small
and medium guard exchanges.

- Small-halo `FillBoundary` on one node: 1.4–2.0x faster.
- Small-halo `FillBoundary` across two nodes: 3–13% less time with the CPU-proxied InfiniBand
  transport and the final prototype. An earlier version with a cheaper wait that NVSHMEM 3.7
  does not guarantee reached 11–26%. GPU-initiated InfiniBand gave no gain.
- Large-halo `FillBoundary`: slower with NVSHMEM (15–57% on one node; 10–30% across two nodes
  when both NICs are used, about 2x when all GPUs of a node share one NIC).
- Scalar reductions: slower than a device-to-host copy plus `MPI_Allreduce`.
- Particle `Redistribute`: nothing to gain from swapping the transport.
- Against NCCL: same speed on a node, simpler to integrate, much better for large halos across
  nodes and no failure during an exchange. For small halos across nodes NCCL's send and receive
  was faster in two of three cases. If AMReX builds a GPU-initiated fast path, NVSHMEM is the
  better base.

## Setup

| | |
|---|---|
| Nodes | 4x NVIDIA GB200 (189 GB each) and 144 Arm cores in two sockets per node |
| GPU links | NVLink between the 4 GPUs of a node; no multi-node NVLink fabric reported |
| Network | InfiniBand, 2x 400 Gb/s per node (one NIC per GPU pair: `mlx5_0` and `mlx5_3`) |
| Software | NVHPC 26.9, CUDA 13.3, NVSHMEM 3.7.2 (bundled; newest release is 3.8.0), HPC-X 2.50 (Open MPI 5.0.10, UCX 1.21), GCC 14.4.1 |
| AMReX | `development` at 1dfb2b721f plus the prototype of this branch; CUDA, MPI, double precision, 3D |
| Runs | one MPI rank per GPU, 8 cores per rank, ranks bound to the GPU's socket |

All MPI runs use `UCX_MEMTYPE_CACHE=n` (see the NCCL summary). The MPI numbers come from the same
binary with `amrex.nvshmem.fb_mode=0`, in which NVSHMEM is not initialized.

## The two NVSHMEM paths

Both reuse the message layout and the kernels of unpack-on-arrival, queue all work on the GPU
stream and synchronize it once per call.

- **Puts** (`amrex.nvshmem.fb_mode=1`): pack into a local send window, then one kernel issues
  `nvshmemx_putmem_signal_block` for every neighbor rank, one thread block per message.
- **Hybrid** (`amrex.nvshmem.fb_mode=2`): where `nvshmem_ptr` maps the receiving GPU (NVLink),
  the pack kernel writes straight into the receiver's window; all other ranks get puts. GROMACS
  ships the same design.

Flags in symmetric memory replace the MPI handshake: a receiver marks its window free
(`nvshmem_uint64_p`), the sender waits for that, writes or puts, and sets a done flag; the
receiver waits for it before the unpack kernel runs.

Two versions of that last wait were measured:

- **Signal wait** (earlier version): `nvshmem_signal_wait_until`. In NVSHMEM 3.7 it can return
  before the data of a network put is visible to the kernel (NVSHMEM issue 75, fixed in 3.8.0).
  No wrong cell was seen with it.
- **Safe wait** (final version, the code on this branch): `nvshmem_uint64_wait_until`, which
  includes the missing step. It costs 1–2 µs per call on one node and about 10 µs across nodes
  (see the micro-benchmark).

## FillBoundary in AMReX

Microseconds per `FillBoundary` call of one `MultiFab` on a periodic domain. Every run passed a
check of all ghost cells (0 wrong cells).

### One node, 4 ranks

| Case | MPI | NVSHMEM puts | NVSHMEM hybrid | NCCL peer writes (earlier study) |
|---|---|---|---|---|
| 128³, 64³ boxes, 1 comp, 1 ghost | 64 | 40 | 33 | 30 |
| 256³, 128³ boxes, 1 comp, 1 ghost | 65–75 | 55 | 37 | 37 |
| 256³, 512 boxes of 32³, 1 comp, 1 ghost | 68 | 67 | 49 | 46 |
| 512³, 256³ boxes, 3 comps, 2 ghosts | 170 | 500 | 196 | 192 |
| 512³ nodal, 128³ boxes, 6 comps, 4 ghosts | 549 | 3829 | 863 | 856 |

Safe wait. All three peers of a rank are NVLink-mapped here, so "hybrid" is direct writes only,
and it equals the NCCL peer-write path: both sit on the same NVLink mapping. With the signal
wait every NVSHMEM number was 1–2 µs lower. The MPI baseline of the second case was 65 µs in the
NCCL study and 75 µs in this one.

### Two nodes, 8 ranks, all ranks of a node on one NIC, signal wait

`NVSHMEM_HCA_LIST=mlx5_0:1`. In hybrid mode each rank writes directly to 3 peers and puts to the
others over InfiniBand.

| Case | MPI | Puts, GPU-initiated IB | Hybrid, GPU-initiated IB | Hybrid, CPU-proxied IB |
|---|---|---|---|---|
| 128³, 64³ boxes, 1 comp, 1 ghost | 91 | 89 | 93 | 67 |
| 256³, 128³ boxes, 1 comp, 1 ghost | 98 | 101 | 104 | 82 |
| 256³, 512 boxes of 32³, 1 comp, 1 ghost | 106 | 110 | 116 | 94 |
| 512³, 256³ boxes, 3 comps, 2 ghosts | 340 | 662 | 696 | 667 |
| 512³ nodal, 128³ boxes, 6 comps, 4 ghosts | 1450 | 2903 | 3105 | 3031 |

"CPU-proxied IB" is NVSHMEM's default `ibrc` transport, where a host proxy thread issues the
network operations for the kernel. "GPU-initiated IB" is IBGDA, where the kernel drives the NIC.

### Two nodes, 8 ranks, one NIC per GPU pair, safe wait

`NVSHMEM_ENABLE_NIC_PE_MAPPING=1 NVSHMEM_HCA_PE_MAPPING=mlx5_0:1:2,mlx5_3:1:2`.

| Case | MPI | Puts, CPU-proxied IB | Hybrid, CPU-proxied IB | Hybrid, GPU-initiated IB |
|---|---|---|---|---|
| 128³, 64³ boxes, 1 comp, 1 ghost | 90 | 78 | 83 | 102 |
| 256³, 128³ boxes, 1 comp, 1 ghost | 98 | 86 | 90 | 108 |
| 256³, 512 boxes of 32³, 1 comp, 1 ghost | 106 | 99 | 103 | 116 |
| 512³, 256³ boxes, 3 comps, 2 ghosts | 342 | 376 | 417 | 445 |
| 512³ nodal, 128³ boxes, 6 comps, 4 ghosts | 1507 | 1952 | 1792 | 1829 |

### Reading the tables

- **Where the gain comes from.** The MPI path is pack, stream synchronization, post and wait,
  unpack, stream synchronization. The NVSHMEM paths queue everything on the GPU stream and
  synchronize once. In the hybrid one-node run of the 256³ case, 24 of 37 µs are that single
  synchronization (waiting for the queued GPU work) and 7 µs launch the ready, pack and done
  kernels.
- **Why kernel-issued puts lose for large halos on a node.** A put over NVLink is a copy by the
  512 threads of one block per message, after a pack into the send window. That is 3–7x slower
  than MPI for megabyte messages. The hybrid path avoids it but still loses 15–57% to UCX's bulk
  GPU-to-GPU copy with unpack-on-arrival, as the NCCL peer writes did.
- **Across nodes, the CPU-proxied transport wins for small halos.** GPU-initiated InfiniBand
  gave no gain in any small-halo case (from 1% less to 14% more time than MPI). NVSHMEM's
  maintainers say its single-message latency is expected to be longer than the proxied
  transport's; its strength is message rate.
- **The two two-node tables differ in two things.** The first has one NIC and the signal wait,
  the second both NICs and the safe wait. The micro-benchmark separates the effects: the second
  NIC halves the time for megabyte messages and does not slow small ones, and the safe wait
  adds about 10 µs. So the small halos most likely lost part of their gain (67–94 µs to
  83–103 µs) to the wait and not to the NICs, and the large halos improved (2x slower than MPI
  to 10–30% slower) because of the NICs. No AMReX run combined both NICs with the signal wait.
- **Puts against hybrid across nodes.** With both NICs, puts alone were faster than the hybrid
  path in four of five cases. This contradicts the one-node result and was not investigated.
- **A real implementation needs a switch by message size**, as for NCCL.

Runs are single measurements of 300–1000 calls. Repeats differed by up to about 10%. The
prototype is untuned.

## Transport micro-benchmark

Every rank exchanges one message with each peer, as a `FillBoundary` does: pack kernel, transfer,
unpack kernel. Microseconds per exchange, maximum over ranks, best of the repetitions. The
kernel-issued and peer-write variants include the flag handshake described above, with the
signal wait unless a row says otherwise.

### One node, 4 ranks, 3 NVLink peers per rank

| Chain | 4 KiB | 256 KiB | 2 MiB | 16 MiB |
|---|---|---|---|---|
| pack and unpack only, no transfer | 12 | 13 | 14 | 47 |
| MPI | 52 | 55 | 68 | 162 |
| host calls on the stream (`nvshmemx_putmem_signal_on_stream`) | 50 | 54 | 83 | 290 |
| kernel-issued `nvshmemx_putmem_signal_block`, 512 threads | 24 | 33 | 89 | 861 |
| the same, 1024 threads | | | 69 | 500 |
| peer writes through `nvshmem_ptr` into a window | 22 | 24 | 42 | 168 |
| peer writes straight into the destination array | 19 | 21 | 38 | 140 |

Transfer alone, without pack and unpack: MPI 30, 32, 44, 102 µs; host calls on the stream 35, 45,
78, 245 µs; kernel-issued puts 19, 26, 70, 774 µs.

### Two nodes, 2 ranks, 1 InfiniBand peer per rank

| Chain | 4 KiB | 256 KiB | 2 MiB | 16 MiB |
|---|---|---|---|---|
| MPI | 31 | 67 | 119 | 589 |
| host calls on the stream, `ibrc` | 28 | 35 | 74 | 387 |
| kernel-issued puts, `ibrc` | 33 | 40 | 79 | 392 |
| kernel-issued puts, `gpunetio` | 37 | 43 | 83 | 402 |
| kernel-issued puts, IBGDA | 46 | 52 | 93 | 412 |

Transfer alone at 4 KiB: MPI 10 µs (eager), NVSHMEM 23–42 µs. So across InfiniBand NVSHMEM is
1.3–1.9x faster from 256 KiB to 16 MiB. For tiny messages the full chain is about equal with
`ibrc` and slower with the GPU-initiated transports. NCCL was in the same range on this link
(34–46, 74–93 and 386–411 µs at the three larger sizes).

### Two nodes, 8 ranks, 7 peers per rank (3 NVLink, 4 InfiniBand)

All seven peers get kernel-issued puts.

| Chain | 4 KiB | 256 KiB | 2 MiB | 16 MiB |
|---|---|---|---|---|
| MPI | 74 | 135 | 694–715 | 3785–4025 |
| puts, `ibrc`, one NIC per node, signal wait | 61 | 142 | 754 | 5700 |
| puts, IBGDA, one NIC per node, signal wait | 85 | 165 | 792 | 5859 |
| puts, IBGDA, one NIC per GPU pair, signal wait | 83 | 123 | 438 | 3017 |
| puts, IBGDA, one NIC per GPU pair, safe wait | 95 | 133 | 450 | 3027 |
| puts, `ibrc`, one NIC per GPU pair, safe wait | 76 | 122 | 430 | 2945 |

- **Second NIC.** The two IBGDA rows with the signal wait differ only in the NIC mapping: no
  change at 4 KiB, 1.3–1.9x faster from 256 KiB on.
- **Safe wait.** The two IBGDA rows with one NIC per GPU pair differ only in the wait: 10–12 µs
  more at every size. They come from two jobs on the same pair of nodes, whose MPI and host-call
  results agreed within 1%, so this is not run-to-run noise.
- **Against MPI.** With both NICs the puts beat MPI by 1.3–1.6x at 2 and 16 MiB. The AMReX
  large-halo cases do not show that gain; the reason was not investigated.

### Collectives

One double per rank, summed over all ranks and returned to the host.

| | 1 node, 4 ranks | 2 nodes, 2 ranks | 2 nodes, 8 ranks |
|---|---|---|---|
| device-to-host copy and `MPI_Allreduce` | 16 | 14 | 20 |
| `nvshmemx_double_sum_reduce_on_stream` | 26 | 27–50 | 40–110 |

Across nodes the lower NVSHMEM number is with `ibrc` and the higher with IBGDA.

### Start-up cost of NVSHMEM

`nvshmemx_init_attr` took 0.5 s for 2 ranks, 0.85 s for 4 and 1.0 s for 8. `nvshmem_malloc` of the
windows took less than a millisecond. NCCL needed 1.3–2.4 s for the communicator plus up to
0.7 s for window registration.

## Particles and collectives

- **Redistribute.** Not measured again. The profile of the NCCL study stands: about 7–8 ms per
  call for 8.4 million particles per GPU on one node, of which the size handshake is 2.1 ms,
  waiting for the particle messages 2.3 ms, partitioning 1.4 ms, pack and post 0.9 ms. The
  handshake and the waits absorb waiting for slower peers, so a faster transport gains a few
  percent at most.
- **One-sided Redistribute.** NVSHMEM could remove the size handshake, which NCCL's send and
  receive could not: senders can put into a peer window without a matching receive. The
  published recipe (PetscSF) sizes the windows once by a global maximum and adds a flow-control
  flag; remote atomics could reserve space instead. Neither was prototyped, and remote atomics
  were not tested on this stack.
- **Reductions.** See the table above: no gain. AMReX's reductions act on host scalars after a
  GPU reduction, so there is little for a GPU-side collective to remove.

## Compared with NCCL

Microseconds per `FillBoundary`; NCCL numbers from `NCCL_Experiment_Summary.md`. The NVSHMEM
numbers are with the safe wait and, on two nodes, the CPU-proxied transport on both NICs.

| | MPI | NCCL `ncclSend`/`ncclRecv` | NCCL peer writes | NVSHMEM hybrid | NVSHMEM puts |
|---|---|---|---|---|---|
| One node, 128³ | 64 | 33 | 30 | 33 | 40 |
| One node, 256³ | 65–75 | 40 | 37 | 37 | 55 |
| One node, 256³ in 512 boxes | 68 | 50 | 46 | 49 | 67 |
| One node, 512³ 3 comps | 170 | 190 | 192 | 196 | 500 |
| One node, 512³ nodal 6 comps | 549 | 733 | 856 | 863 | 3829 |
| Two nodes, 128³ | 90 | 68 | not possible | 83 | 78 |
| Two nodes, 256³ | 97–98 | 91 | not possible | 90 | 86 |
| Two nodes, 256³ in 512 boxes | 105–106 | 86 | not possible | 103 | 99 |
| Two nodes, 512³ 3 comps | 340–342 | 624 | not possible | 417 | 376 |
| Two nodes, 512³ nodal 6 comps | 1450–1600 | crashed | not possible | 1792 | 1952 |

With the signal wait and one NIC, NVSHMEM's hybrid path took 67, 82 and 94 µs in the three
small two-node cases.

| | NCCL 2.31 | NVSHMEM 3.7 |
|---|---|---|
| Memory for peer access | windows from `ncclMemAlloc`, registered collectively | symmetric heap, `nvshmem_malloc` |
| Off-node path from a kernel | separate API (GIN) with its own device communicator | the same put call as on the node |
| Matching | send and receive must match in size and type | one-sided, no receive call |
| Remote atomics | none in GIN | in the API; not tested here, and documented to need GDRCopy with the proxied transports |
| Sub-communicators | yes | no, one world |
| Start-up | 1.3–2.4 s and window registration | 0.5–1.0 s |
| Build with AMReX | links against `libnccl` | compiled and linked with AMReX's default CUDA build (relocatable device code), static device library |
| Failures inside AMReX runs | segmentation fault in the InfiniBand plugin for one large case | none during the exchange; crash in finalize with NIC mapping |

## Problems found on this stack

- **The default NIC choice crashes.** NVSHMEM picks the Ethernet device `mlx5_1` and fails during
  initialization. Set `NVSHMEM_HCA_LIST=mlx5_0:1` or the NIC mapping above, or
  `NVSHMEM_REMOTE_TRANSPORT=none` on one node. MPI and UCX device settings are not honoured.
- **GPU-initiated InfiniBand is off by default.** It needs `NVSHMEM_IB_ENABLE_IBGDA=1`.
- **The `ucx` transport aborts** when a signal is used ("AMO 8 not supported", it needs GDRCopy).
- **Finalize crashes with the NIC mapping.** With `NVSHMEM_ENABLE_NIC_PE_MAPPING=1`, NVSHMEM
  segfaults in `nvshmemid_hostlib_finalize` (`ibv_dealloc_pd`) for both InfiniBand transports,
  after all results are printed. AMReX's signal handler turns this into a hang of minutes; run
  with `amrex.signal_handling=0` until it is fixed. Worth reporting to NVIDIA.
- **Device-side signal wait in 3.7.** `nvshmem_signal_wait_until` in a kernel can return before
  the data of a network put is visible to that kernel (NVSHMEM issue 75, fixed in 3.8.0). No
  wrong cell was seen with it, but the prototype now waits with `nvshmem_uint64_wait_until`,
  which includes the missing step and costs about 10 µs per exchange across nodes. Whether the
  fixed signal wait of 3.8.0 costs the same was not measured.
- **Library path.** `LD_LIBRARY_PATH` must contain NVSHMEM's `lib` directory, because the
  bootstrap and transport plugins are loaded at run time.

## NVSHMEM constraints for AMReX

- One MPI rank per GPU; host and stream calls from one thread at a time.
- One world: NVSHMEM is initialized once for a fixed set of ranks, there are no
  sub-communicators, and every rank connects to every rank at initialization. NVIDIA's best
  practice notes say the reliable-connection queue pairs "can be prohibitive for large-scale
  applications"; the largest published run is 1152 GPUs (GROMACS).
- Symmetric allocation is collective over all ranks, so windows must be sized up front
  (`NVSHMEM_SYMMETRIC_SIZE`, `amrex.nvshmem.window_mb`) and cannot follow a regrid cheaply.
- Destinations must be in the symmetric heap, so AMReX arena buffers cannot be targets.
  According to NVSHMEM's FAQ, passing heap buffers to a GPU-aware MPI can crash with the default
  heap kind (CUDA VMM).
- The CPU-proxied transports keep a busy-polling proxy thread per rank, which needs a free
  core. GROMACS reports up to 50x slowdown when it shares a busy core.
- `nvshmem_ptr` is only non-null for GPUs reachable by loads and stores: the 4 GPUs of a node
  here, a whole NVLink domain on multi-node NVLink systems.

## Prior art

- **GROMACS** (NVSHMEM 3.1/3.2, H100, InfiniBand): up to 1.5x on one node and up to 1.3x across
  nodes against GPU-aware MPI, with the hybrid design used here; "for larger systems at low node
  counts, MPI marginally outperforms NVSHMEM". Its manual calls the feature experimental and
  best for small systems. This matches the small-halo and large-halo split measured here.
- **PetscSF** (NVSHMEM 2.0.3, V100): small-message latency 12 µs higher than MPI on a node and
  40 µs higher across nodes; world communicator only; extra copies into symmetric buffers.
- **QUDA** reports about 20% less solver time at 256 GPUs (slides only).
- No use in an AMR or particle-in-cell framework was found.

## Recommendation

Do not build a general NVSHMEM backend. The worthwhile piece is the same as after the NCCL
study: an opt-in fast path for latency-bound `FillBoundary` and `SumBoundary` (multigrid coarse
levels, small boxes) with a size threshold that falls back to MPI. Build it on NVSHMEM rather
than NCCL: one mechanism covers NVLink and InfiniBand, it needs no receive-side call, and it
was the more robust of the two in these runs. Use the CPU-proxied transport across nodes; the
GPU-initiated one does not help a halo exchange with a handful of messages per call.

The gain across nodes is small with the safe wait (3–13%), so the case for the fast path rests
on single-node and NVLink-domain runs until the wait cost is understood.

## The prototype on this branch

Experimental and opt-in. Configure with `-DAMReX_NVSHMEM=ON -DNVSHMEM_ROOT=<nvshmem prefix>`
(needs CUDA and MPI), then choose the path at run time:

| Input | Meaning |
|---|---|
| `amrex.nvshmem.fb_mode=0` | MPI (default); NVSHMEM is not initialized |
| `amrex.nvshmem.fb_mode=1` | pack into a send window, kernel-issued puts with signals to every neighbor rank |
| `amrex.nvshmem.fb_mode=2` | hybrid: direct writes where `nvshmem_ptr` maps the receiver, puts elsewhere |
| `amrex.nvshmem.window_mb` | size of each rank's send and receive window (default 256) |

The symmetric heap must hold both windows: set the environment variable `NVSHMEM_SYMMETRIC_SIZE`
accordingly (the runs here used 8G with 1024 MB windows).

Files: `Src/Base/AMReX_NVSHMEM.{H,cpp}` (initialization, windows, flags),
`Src/Base/AMReX_FBNvshmem.H` (the paths), hooks in `AMReX_FabArrayCommI.H`, `AMReX_FabArray.H`
and `AMReX.cpp`.

Limits:

- Only `FillBoundary` and `SumBoundary` on GPU, with the same conditions as unpack-on-arrival
  (not deterministic, no CUDA graph). `ParallelCopy` and particles still use MPI.
- No switch by message size, so large halos get slower.
- One `FillBoundary` in flight per rank and at most 32 neighbor ranks.
- CMake only; the NVSHMEM dependency is not exported to installed CMake configs. No
  documentation, no tests.

## Open items

- NVSHMEM 3.8.0: fixed signal wait, multi-NIC support in the proxied transport, batched
  non-blocking puts.
- Where the 10 µs of the safe wait across nodes go, and whether the fixed signal wait of 3.8.0
  avoids them.
- An AMReX run with both NICs and the signal wait, to separate the two effects directly.
- Why the micro-benchmark's gain for megabyte messages with both NICs does not reach the AMReX
  large-halo cases, and why puts alone matched the hybrid path across nodes.
- A put kernel with several thread blocks per message, and NVSHMEM's TMA-backed NVLink copies
  (`NVSHMEM_TMA_POLICY`, off by default), for large messages on a node.
- A system whose NVLink domain spans nodes, where the hybrid path would write directly to all
  peers.
- Several exchanges in flight, a size threshold with MPI fallback, `ParallelCopy`.
- One-sided `Redistribute` without the size handshake.
- Report the finalize crash and the default NIC choice.

## Where the runs are

On cech in `~/2026-11-25_nvshmem_dbdf7ada`: the micro-benchmark `bench/bench_shmem.cu`, the
`FillBoundary` benchmark `amrex/Tests/FillBoundaryBench`, the build `build-shm`, the step
scripts and all logs (`runs/`, `runs-2node/`, `runs-2nic/`, `runs-2nic2/`).
