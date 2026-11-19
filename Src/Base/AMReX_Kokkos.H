#ifndef AMREX_KOKKOS_H_
#define AMREX_KOKKOS_H_
#include <AMReX_Config.H>

/**
 * \file AMReX_Kokkos.H
 * \brief Kokkos as AMReX's on-node performance-portability layer.
 *
 * In GPU builds, AMReX launches its kernels, reductions and scans through
 * Kokkos, allocates device memory through Kokkos memory spaces, and maps
 * each AMReX GPU stream to a Kokkos execution space instance.  AMReX's
 * default stream (stream index 0) is the default instance of
 * Kokkos::DefaultExecutionSpace, so work that user code submits to
 * Kokkos::DefaultExecutionSpace() is ordered with AMReX's default stream.
 *
 * In CPU builds, AMReX initializes Kokkos (if the application has not
 * done so already) so that applications can use Kokkos together with
 * AMReX, but AMReX's host loops remain plain loop nests.
 */

#if defined(AMREX_USE_GPU) && !defined(AMREX_USE_KOKKOS)
#  error "AMReX GPU builds require Kokkos (AMREX_USE_KOKKOS)."
#endif

#ifdef AMREX_USE_KOKKOS

#include <Kokkos_Macros.hpp>

// Kokkos' atomics require the same relocatable device code mode in Kokkos and AMReX.
#if defined(KOKKOS_ENABLE_CUDA) && defined(__CUDACC__)
#  if defined(KOKKOS_ENABLE_CUDA_RELOCATABLE_DEVICE_CODE) && !defined(__CUDACC_RDC__)
#    error "Kokkos was built with Kokkos_ENABLE_CUDA_RELOCATABLE_DEVICE_CODE=ON, but AMReX is compiled without relocatable device code. Build AMReX with AMReX_GPU_RDC=ON (CMake) or USE_GPU_RDC=TRUE (GNU make), or use a Kokkos built without it."
#  elif !defined(KOKKOS_ENABLE_CUDA_RELOCATABLE_DEVICE_CODE) && defined(__CUDACC_RDC__)
#    error "AMReX is compiled with relocatable device code, but Kokkos was built without Kokkos_ENABLE_CUDA_RELOCATABLE_DEVICE_CODE. Build AMReX with AMReX_GPU_RDC=OFF (CMake) or USE_GPU_RDC=FALSE (GNU make), or use a Kokkos built with it."
#  endif
#endif

#include <Kokkos_Core.hpp>

#include <cstddef>
#include <string>

#if defined(AMREX_USE_CUDA) && !defined(KOKKOS_ENABLE_CUDA)
#  error "AMReX was configured for CUDA, but Kokkos was built without Kokkos_ENABLE_CUDA."
#endif
#if defined(AMREX_USE_HIP) && !defined(KOKKOS_ENABLE_HIP)
#  error "AMReX was configured for HIP, but Kokkos was built without Kokkos_ENABLE_HIP."
#endif
#if defined(AMREX_USE_SYCL) && !defined(KOKKOS_ENABLE_SYCL)
#  error "AMReX was configured for SYCL, but Kokkos was built without Kokkos_ENABLE_SYCL."
#endif
#if !defined(AMREX_USE_GPU) && (defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP) || defined(KOKKOS_ENABLE_SYCL))
#  error "AMReX was configured without a GPU backend, but Kokkos was built with a device backend."
#endif
#if !defined(KOKKOS_ENABLE_SERIAL)
#  error "AMReX requires Kokkos to be built with Kokkos_ENABLE_SERIAL."
#endif
#if defined(AMREX_USE_OMP) && !defined(KOKKOS_ENABLE_OPENMP)
#  error "AMReX was configured with OpenMP, but Kokkos was built without Kokkos_ENABLE_OPENMP."
#endif

namespace amrex::Gpu {

#ifdef AMREX_USE_GPU
//! Kokkos execution space for AMReX device kernels
using KokkosExecSpace = Kokkos::DefaultExecutionSpace;
#else
//! Kokkos execution space for AMReX (serial, per-thread) host kernels
using KokkosExecSpace = Kokkos::Serial;
#endif

//! Kokkos memory space for AMReX device memory (The_Arena)
using KokkosDeviceMemSpace = KokkosExecSpace::memory_space;

}

namespace amrex::kokkos {

/**
 * \brief Initialize Kokkos, unless the application has done so already.
 *
 * Called by amrex::Initialize.  If AMReX initializes Kokkos, it uses
 * GPU `device_id` (if non-negative) and AMReX's number of OpenMP threads,
 * and it finalizes Kokkos at program exit, because Kokkos cannot be
 * re-initialized after Kokkos::finalize while AMReX can.  ParmParse
 * parameter `amrex.kokkos_disable_warnings` (default: true) controls
 * Kokkos' initialization warnings.
 */
void Initialize (int device_id);

//! Called by amrex::Finalize.  Fences Kokkos; it does not finalize it.
void Finalize ();

//! Whether AMReX (rather than the application) initialized Kokkos
[[nodiscard]] bool InitializedByAMReX () noexcept;

#ifdef AMREX_USE_GPU
//! Kind of GPU-accessible memory, i.e., Kokkos memory space
enum class MemoryKind {
    Device,  //!< Gpu::KokkosDeviceMemSpace (e.g., Kokkos::CudaSpace)
    Managed, //!< Kokkos::SharedSpace (e.g., Kokkos::CudaUVMSpace)
    Pinned   //!< Kokkos::SharedHostPinnedSpace (e.g., Kokkos::CudaHostPinnedSpace)
};

/**
 * \brief Allocate `nbytes` bytes from a Kokkos memory space.
 *
 * Kokkos Tools see the allocation with the given label.  If the allocation
 * fails, this returns nullptr and stores Kokkos' error message in `error`
 * (if not null).
 */
[[nodiscard]] void* allocate (MemoryKind kind, std::size_t nbytes, char const* label,
                              std::string* error = nullptr);

//! Deallocate memory obtained from kokkos::allocate.
void deallocate (MemoryKind kind, void* p, std::size_t nbytes, char const* label);
#endif

}

#endif
#endif
