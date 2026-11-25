#ifndef AMREX_NCCL_H_
#define AMREX_NCCL_H_
#include <AMReX_Config.H>

/**
 * \file AMReX_NCCL.H
 * \brief Experimental: an NVIDIA NCCL communicator next to MPI, used by the
 *        optional NCCL FillBoundary paths.
 *
 * Select a path at run time with `amrex.nccl.fb_mode`:
 *  - 0: MPI (default)
 *  - 1: `ncclSend`/`ncclRecv` on the current GPU stream
 *  - 2: pack kernels write straight into the receiving GPU over NVLink
 *       (needs all ranks in one NVLink load/store domain)
 *
 * NCCL needs one MPI rank per GPU.
 */

#ifdef AMREX_USE_NCCL

#include <nccl.h>

#include <cstddef>

namespace amrex::NCCL {

//! FillBoundary communication path.
enum struct FBMode : int { off = 0, sendrecv = 1, lsa = 2 };

//! Create the NCCL communicator if `amrex.nccl.fb_mode` asks for one. Collective.
void Initialize ();
//! Release the communicator and its windows. Collective.
void Finalize ();

//! The FillBoundary path in use.
[[nodiscard]] FBMode fbMode () noexcept;
//! The NCCL communicator; its ranks are the MPI ranks.
[[nodiscard]] ncclComm_t comm () noexcept;

/// \cond DOXYGEN_IGNORE
// Mode 2: every rank owns a receive window (`amrex.nccl.window_mb`) and a flag
// array that its peers can write through NVLink. Pair counters number the
// messages between two ranks; flags carry those numbers.
[[nodiscard]] char* recvWindow () noexcept;
[[nodiscard]] std::size_t recvWindowSize () noexcept;
[[nodiscard]] char* peerRecvWindow (int rank) noexcept;
[[nodiscard]] const unsigned long long* doneFlags () noexcept;   // [r]: messages rank r completed here
[[nodiscard]] const unsigned long long* readyFlags () noexcept;  // [r]: messages rank r is ready to take
[[nodiscard]] unsigned long long* peerDoneFlag (int rank) noexcept;
[[nodiscard]] unsigned long long* peerReadyFlag (int rank) noexcept;
[[nodiscard]] unsigned long long nextSent (int rank) noexcept;
[[nodiscard]] unsigned long long nextExpected (int rank) noexcept;
//! Number of mode 2 exchanges this rank has started and not finished.
[[nodiscard]] int& lsaInFlight () noexcept;
//! Whether peer-writing kernels fence their stores (`amrex.nccl.fence`, for experiments only).
[[nodiscard]] bool fence () noexcept;

void check (ncclResult_t r, const char* what);
/// \endcond

}

#endif
#endif
