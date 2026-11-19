#ifndef AMREX_GPU_TYPES_H_
#define AMREX_GPU_TYPES_H_
#include <AMReX_Config.H>

#ifdef AMREX_USE_GPU

#include <AMReX_Extension.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_Kokkos.H>

#ifdef AMREX_USE_SYCL
#  include <sycl/sycl.hpp>
#  include <cstddef>
#endif

namespace amrex {

#ifdef AMREX_USE_SYCL

struct dim3 {
    unsigned int x = 1;
    unsigned int y = 1;
    unsigned int z = 1;
    dim3 () = default;
    constexpr dim3 (unsigned int x_, unsigned int y_=1, unsigned int z_=1) : x(x_),y(y_),z(z_) {}
};


struct gpuStream_t {
    sycl::queue* queue = nullptr;
    bool operator== (gpuStream_t const& rhs) const noexcept { return queue == rhs.queue; }
    bool operator!= (gpuStream_t const& rhs) const noexcept { return queue != rhs.queue; }
};

#endif

}

namespace amrex {
//! 1D block or thread index, for kernels written without CUDA/HIP built-in variables
struct Dim1 {
    std::size_t x;
};
}

#endif

namespace amrex::Gpu {

#ifdef AMREX_USE_GPU
//! Kokkos team (the AMReX "block") of a ParallelFor kernel
using KokkosTeamMember = Kokkos::TeamPolicy<KokkosExecSpace>::member_type;
#endif

#if defined(AMREX_USE_SYCL)

struct Handler
{
    AMREX_GPU_HOST_DEVICE constexpr
    Handler (sycl::nd_item<1> const* a_item = nullptr, void* a_local = nullptr,
             int a_n_active_threds = -1)
        : item(a_item), local(a_local), numActiveThreads(a_n_active_threds) {}

    //! Handler of a Kokkos-launched kernel (Kokkos team instead of a SYCL nd_item)
    AMREX_GPU_HOST_DEVICE constexpr
    Handler (int a_n_active_threads, KokkosTeamMember const* a_team, void* a_local = nullptr)
        : item(nullptr), local(a_local), numActiveThreads(a_n_active_threads), team(a_team) {}

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    bool isFullBlock () const {
        if (team) {
            return (numActiveThreads >= team->team_size() ||
                    numActiveThreads <= 0);
        }
        return (numActiveThreads >= int(item->get_local_range(0)) ||
                numActiveThreads <= 0);
    }

    std::size_t globalIdx () const { return team ? blockIdx()*blockDim() + threadIdx() : item->get_global_linear_id(); }
    std::size_t blockIdx () const { return team ? std::size_t(team->league_rank()) : item->get_group_linear_id(); }
    std::size_t threadIdx () const { return team ? std::size_t(team->team_rank()) : item->get_local_linear_id(); }
    //
    std::size_t gridDim () const { return team ? std::size_t(team->league_size()) : item->get_group_range(0); }
    std::size_t blockDim () const { return team ? std::size_t(team->team_size()) : item->get_local_range(0); }

    // warp index in block
    std::size_t warpIdx () const { return item->get_sub_group().get_group_id()[0]; }
    // lane index in warp
    std::size_t laneIdx () const { return item->get_sub_group().get_local_id()[0]; }
    // warp size
    std::size_t warpDim () const { return item->get_sub_group().get_local_range()[0]; }

    void* sharedMemory () const { return local; }

    void syncThreads () const {
        if (team) { team->team_barrier(); } else { sycl::group_barrier(item->get_group()); }
    }

    sycl::nd_item<1> const* item;
    void* local; // SYCL shared local memory
    int numActiveThreads;
    KokkosTeamMember const* team = nullptr;
};

#elif defined(AMREX_USE_GPU)

/**
 * \brief Block (Kokkos team) information passed to ParallelFor lambdas that
 * take a Gpu::Handler, for block-level operations such as Gpu::deviceReduceSum.
 */
struct Handler
{
    AMREX_GPU_HOST_DEVICE constexpr explicit
    Handler (int n_active_threads = -1, KokkosTeamMember const* a_team = nullptr,
             void* a_local = nullptr)
        : numActiveThreads(n_active_threads), team(a_team), local(a_local) {}

    //! Whether all threads of the block (team) are active
    AMREX_GPU_DEVICE AMREX_FORCE_INLINE
    bool isFullBlock () const {
        return (numActiveThreads >= team->team_size() ||
                numActiveThreads <= 0);
    }

    //! Global thread index
    AMREX_GPU_DEVICE AMREX_FORCE_INLINE
    std::size_t globalIdx () const { return blockIdx()*blockDim() + threadIdx(); }
    //! Block (team) index
    AMREX_GPU_DEVICE AMREX_FORCE_INLINE
    std::size_t blockIdx () const { return std::size_t(team->league_rank()); }
    //! Thread index in the block (team)
    AMREX_GPU_DEVICE AMREX_FORCE_INLINE
    std::size_t threadIdx () const { return std::size_t(team->team_rank()); }
    //! Number of blocks (teams)
    AMREX_GPU_DEVICE AMREX_FORCE_INLINE
    std::size_t gridDim () const { return std::size_t(team->league_size()); }
    //! Number of threads in a block (team)
    AMREX_GPU_DEVICE AMREX_FORCE_INLINE
    std::size_t blockDim () const { return std::size_t(team->team_size()); }

    //! Shared memory (team scratch) of the block, see Gpu::handler_launch
    AMREX_GPU_DEVICE AMREX_FORCE_INLINE
    void* sharedMemory () const { return local; }

    //! Barrier for all threads of the block (team)
    AMREX_GPU_DEVICE AMREX_FORCE_INLINE
    void syncThreads () const { team->team_barrier(); }

    int numActiveThreads;
    KokkosTeamMember const* team;
    void* local; //!< shared memory (team scratch)
};

#else

struct Handler {};

#endif

}

#endif
