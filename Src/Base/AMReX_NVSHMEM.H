#ifndef AMREX_NVSHMEM_H_
#define AMREX_NVSHMEM_H_
#include <AMReX_Config.H>

/**
 * \file AMReX_NVSHMEM.H
 * \brief Experimental: NVIDIA NVSHMEM next to MPI, used by the optional
 *        NVSHMEM FillBoundary paths.
 *
 * Select a path at run time with `amrex.nvshmem.fb_mode`:
 *  - 0: MPI (default)
 *  - 1: kernels issue NVSHMEM puts with signals for every neighbor rank
 *  - 2: as 1, but pack kernels write straight into the receiving GPU where
 *       NVSHMEM maps it (NVLink), and puts are used for all other ranks
 *
 * NVSHMEM is initialized from the MPI communicator, with one MPI rank per
 * GPU. Each rank owns a send and a receive window of `amrex.nvshmem.window_mb`
 * in the symmetric heap (size it with the environment variable
 * `NVSHMEM_SYMMETRIC_SIZE`).
 */

#ifdef AMREX_USE_NVSHMEM

#include <cstddef>
#include <cstdint>

namespace amrex::NVSHMEM {

//! FillBoundary communication path.
enum struct FBMode : int { off = 0, put = 1, hybrid = 2 };

//! Initialize NVSHMEM if `amrex.nvshmem.fb_mode` asks for it. Collective.
void Initialize ();
//! Release the windows and finalize NVSHMEM. Collective.
void Finalize ();

//! The FillBoundary path in use.
[[nodiscard]] FBMode fbMode () noexcept;

/// \cond DOXYGEN_IGNORE
[[nodiscard]] char* recvWindow () noexcept;
[[nodiscard]] char* sendWindow () noexcept;
[[nodiscard]] std::size_t windowSize () noexcept;
//! The rank's receive window as mapped into this GPU, or nullptr if it is not (or mode 1).
[[nodiscard]] char* peerRecvWindow (int rank) noexcept;
// Symmetric flag arrays, one word per rank. Pair counters number the messages
// between two ranks; flags carry those numbers.
[[nodiscard]] std::uint64_t* doneFlags () noexcept;   // [r]: messages rank r completed here
[[nodiscard]] std::uint64_t* readyFlags () noexcept;  // [r]: messages rank r is ready to take
[[nodiscard]] std::uint64_t nextSent (int rank) noexcept;
[[nodiscard]] std::uint64_t nextExpected (int rank) noexcept;
//! Number of exchanges this rank has started and not finished.
[[nodiscard]] int& inFlight () noexcept;
/// \endcond

}

#endif
#endif
