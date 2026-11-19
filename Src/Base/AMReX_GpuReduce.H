#ifndef AMREX_GPU_REDUCE_H_
#define AMREX_GPU_REDUCE_H_
#include <AMReX_Config.H>

#include <AMReX_GpuQualifiers.H>
#include <AMReX_GpuControl.H>
#include <AMReX_GpuTypes.H>
#include <AMReX_GpuAtomic.H>
#include <AMReX_GpuUtility.H>
#include <AMReX_Functional.H>
#include <AMReX_TypeTraits.H>

#if defined(AMREX_USE_CUDA)
#include <cub/cub.cuh>
#if defined(CCCL_MAJOR_VERSION) && defined(CCCL_MINOR_VERSION) && ((CCCL_MAJOR_VERSION >= 3) || ((CCCL_MAJOR_VERSION >= 2) && (CCCL_MINOR_VERSION >= 8)))
#define AMREX_CUDA_CCCL_VER_GE_2_8 1
#endif
#elif defined(AMREX_USE_HIP)
#include <rocprim/rocprim.hpp>
#endif

//
// Public interface
//
namespace amrex::Gpu {
    template <typename T>
    AMREX_GPU_DEVICE AMREX_FORCE_INLINE
    void deviceReduceSum (T * dest, T source, Gpu::Handler const& h) noexcept;

    template <typename T>
    AMREX_GPU_DEVICE AMREX_FORCE_INLINE
    void deviceReduceMin (T * dest, T source, Gpu::Handler const& h) noexcept;

    template <typename T>
    AMREX_GPU_DEVICE AMREX_FORCE_INLINE
    void deviceReduceMax (T * dest, T source, Gpu::Handler const& h) noexcept;

    AMREX_GPU_DEVICE AMREX_FORCE_INLINE
    void deviceReduceLogicalAnd (int * dest, int source, Gpu::Handler const& h) noexcept;

    AMREX_GPU_DEVICE AMREX_FORCE_INLINE
    void deviceReduceLogicalOr (int * dest, int source, Gpu::Handler const& h) noexcept;
}

//
// Reduce functions based on _shfl_down_sync
//

namespace amrex::Gpu {

#if defined(AMREX_USE_GPU)

// Block-level reductions with a Gpu::Handler, implemented with Kokkos team reductions


/// \cond DOXYGEN_IGNORE
namespace detail {
    // Block (Kokkos team) reduction followed by one atomic update per block.
    // A partially active block (the last one of a kernel) uses one atomic
    // update per active thread, because team_reduce needs the whole team.
    template <typename Reducer, typename AtomicOp, typename T>
    AMREX_GPU_DEVICE AMREX_FORCE_INLINE
    void kokkos_device_reduce (T* dest, T source, Gpu::Handler const& handler) noexcept
    {
        if (handler.isFullBlock()) {
            handler.team->team_reduce(Reducer(source));
            if (handler.team->team_rank() == 0) { AtomicOp{}(dest, source); }
        } else {
            AtomicOp{}(dest, source);
        }
    }
}
/// \endcond

// Block-wide (Kokkos team) reductions; the result is returned to all threads.
// All threads of the block must call them.

template <typename T>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
T blockReduceSum (T source, Gpu::Handler const& h) noexcept
{
    h.team->team_reduce(Kokkos::Sum<T>(source));
    return source;
}

template <typename T>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
T blockReduceMin (T source, Gpu::Handler const& h) noexcept
{
    h.team->team_reduce(Kokkos::Min<T>(source));
    return source;
}

template <typename T>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
T blockReduceMax (T source, Gpu::Handler const& h) noexcept
{
    h.team->team_reduce(Kokkos::Max<T>(source));
    return source;
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
int blockReduceLogicalAnd (int source, Gpu::Handler const& h) noexcept
{
    h.team->team_reduce(Kokkos::LAnd<int>(source));
    return source;
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
int blockReduceLogicalOr (int source, Gpu::Handler const& h) noexcept
{
    h.team->team_reduce(Kokkos::LOr<int>(source));
    return source;
}

// Block-wide reduction followed by one atomic update by thread 0

template <typename T>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void deviceReduceSum_full (T * dest, T source, Gpu::Handler const& h) noexcept
{
    T aggregate = blockReduceSum(source, h);
    if (h.threadIdx() == 0) { Gpu::Atomic::AddNoRet(dest, aggregate); }
}

template <typename T>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void deviceReduceMin_full (T * dest, T source, Gpu::Handler const& h) noexcept
{
    T aggregate = blockReduceMin(source, h);
    if (h.threadIdx() == 0) { Gpu::Atomic::Min(dest, aggregate); }
}

template <typename T>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void deviceReduceMax_full (T * dest, T source, Gpu::Handler const& h) noexcept
{
    T aggregate = blockReduceMax(source, h);
    if (h.threadIdx() == 0) { Gpu::Atomic::Max(dest, aggregate); }
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void deviceReduceLogicalAnd_full (int * dest, int source, Gpu::Handler const& h) noexcept
{
    int aggregate = blockReduceLogicalAnd(source, h);
    if (h.threadIdx() == 0) { Gpu::Atomic::LogicalAnd(dest, aggregate); }
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void deviceReduceLogicalOr_full (int * dest, int source, Gpu::Handler const& h) noexcept
{
    int aggregate = blockReduceLogicalOr(source, h);
    if (h.threadIdx() == 0) { Gpu::Atomic::LogicalOr(dest, aggregate); }
}

template <typename T>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void deviceReduceSum (T * dest, T source, Gpu::Handler const& handler) noexcept
{
    detail::kokkos_device_reduce<Kokkos::Sum<T>, Gpu::AtomicAdd<T>>(dest, source, handler);
}

template <typename T>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void deviceReduceMin (T * dest, T source, Gpu::Handler const& handler) noexcept
{
    detail::kokkos_device_reduce<Kokkos::Min<T>, Gpu::AtomicMin<T>>(dest, source, handler);
}

template <typename T>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void deviceReduceMax (T * dest, T source, Gpu::Handler const& handler) noexcept
{
    detail::kokkos_device_reduce<Kokkos::Max<T>, Gpu::AtomicMax<T>>(dest, source, handler);
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void deviceReduceLogicalAnd (int * dest, int source, Gpu::Handler const& handler) noexcept
{
    detail::kokkos_device_reduce<Kokkos::LAnd<int>, Gpu::AtomicLogicalAnd<int>>(dest, source, handler);
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void deviceReduceLogicalOr (int * dest, int source, Gpu::Handler const& handler) noexcept
{
    detail::kokkos_device_reduce<Kokkos::LOr<int>, Gpu::AtomicLogicalOr<int>>(dest, source, handler);
}

#endif

#if defined(AMREX_USE_CUDA) || defined(AMREX_USE_HIP)

// Block-level reductions for raw CUDA/HIP kernels (amrex::launch with native thread indices)


/// \cond DOXYGEN_IGNORE
namespace detail {

template <typename T>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
T shuffle_down (T x, int offset) noexcept
{
    return AMREX_HIP_OR_CUDA(__shfl_down(x, offset),
                             __shfl_down_sync(0xffffffff, x, offset));
}

// If other sizeof is needed, we can implement it later.
template <class T>
requires (sizeof(T)%sizeof(unsigned int) == 0)
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
T multi_shuffle_down (T x, int offset) noexcept
{
    constexpr int nwords = (sizeof(T) + sizeof(unsigned int) - 1) / sizeof(unsigned int);
    T y;
    auto py = reinterpret_cast<unsigned int*>(&y);
    auto px = reinterpret_cast<unsigned int*>(&x);
    for (int i = 0; i < nwords; ++i) {
        py[i] = shuffle_down(px[i],offset);
    }
    return y;
}

}
/// \endcond

template <int warpSize, typename T, typename F>
struct warpReduce
{
    // Not all arithmetic types can be taken by shuffle_down, but it's good enough.
    AMREX_GPU_DEVICE AMREX_FORCE_INLINE
    T operator()(T x) const noexcept
    requires (std::is_arithmetic_v<T>)
    {
        for (int offset = warpSize/2; offset > 0; offset /= 2) {
            T y = detail::shuffle_down(x, offset);
            x = F()(x,y);
        }
        return x;
    }

    AMREX_GPU_DEVICE AMREX_FORCE_INLINE
    T operator()(T x) const noexcept
    requires (!std::is_arithmetic_v<T>)
    {
        for (int offset = warpSize/2; offset > 0; offset /= 2) {
            T y = detail::multi_shuffle_down(x, offset);
            x = F()(x,y);
        }
        return x;
    }
};

template <int warpSize, typename T, typename WARPREDUCE>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
T blockReduce (T x, WARPREDUCE && warp_reduce, T x0)
{
    __shared__ T shared[warpSize];
    int lane = threadIdx.x % warpSize;
    int wid = threadIdx.x / warpSize;
    x = warp_reduce(x);
    // __syncthreads() prior to writing to shared memory is necessary
    // if this reduction call is occurring multiple times in a kernel,
    // and since we don't know how many times the user is calling it,
    // we do it always to be safe.
    __syncthreads();
    if (lane == 0) { shared[wid] = x; }
    __syncthreads();
    bool b = (threadIdx.x == 0) || (threadIdx.x < blockDim.x / warpSize);
    x =  b ? shared[lane] : x0;
    if (wid == 0) { x = warp_reduce(x); }
    return x;
}

// Compute the block-wide reduction for thread0
template <typename T>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
T blockReduceSum (T source) noexcept
{
    return Gpu::blockReduce<Gpu::Device::warp_size>
        (source, Gpu::warpReduce<Gpu::Device::warp_size,T,amrex::Plus<T> >(), (T)0);
}

template <typename T>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void deviceReduceSum_full (T * dest, T source) noexcept
{
    T aggregate = blockReduceSum(source);
    if (threadIdx.x == 0) { Gpu::Atomic::AddNoRet(dest, aggregate); }
}

// Compute the block-wide reduction for thread0
template <int BLOCKDIMX, typename T>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
T blockReduceSum (T source) noexcept
{
#if defined(AMREX_USE_CUDA)
    using BlockReduce = cub::BlockReduce<T,BLOCKDIMX>;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    // __syncthreads() prior to writing to shared memory is necessary
    // if this reduction call is occurring multiple times in a kernel,
    // and since we don't know how many times the user is calling it,
    // we do it always to be safe.
    __syncthreads();
    return BlockReduce(temp_storage).Sum(source);
#elif defined(AMREX_USE_HIP)
    using BlockReduce = rocprim::block_reduce<T,BLOCKDIMX>;
    __shared__ typename BlockReduce::storage_type temp_storage;
    __syncthreads();
    BlockReduce().reduce(source, source, temp_storage, rocprim::plus<T>());
    return source;
#else
    return blockReduceSum(source);
#endif
}

template <int BLOCKDIMX, typename T>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void deviceReduceSum_full (T * dest, T source) noexcept
{
    T aggregate = blockReduceSum<BLOCKDIMX>(source);
    if (threadIdx.x == 0) { Gpu::Atomic::AddNoRet(dest, aggregate); }
}

// Compute the block-wide reduction for thread0
template <typename T>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
T blockReduceMin (T source) noexcept
{
    return Gpu::blockReduce<Gpu::Device::warp_size>
        (source, Gpu::warpReduce<Gpu::Device::warp_size,T,amrex::Minimum<T> >(), source);
}

template <typename T>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void deviceReduceMin_full (T * dest, T source) noexcept
{
    T aggregate = blockReduceMin(source);
    if (threadIdx.x == 0) { Gpu::Atomic::Min(dest, aggregate); }
}

// Compute the block-wide reduction for thread0
template <int BLOCKDIMX, typename T>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
T blockReduceMin (T source) noexcept
{
#if defined(AMREX_USE_CUDA)
    using BlockReduce = cub::BlockReduce<T,BLOCKDIMX>;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    // __syncthreads() prior to writing to shared memory is necessary
    // if this reduction call is occurring multiple times in a kernel,
    // and since we don't know how many times the user is calling it,
    // we do it always to be safe.
    __syncthreads();

#ifdef AMREX_CUDA_CCCL_VER_GE_2_8
    return BlockReduce(temp_storage).Reduce(source, cuda::minimum<T>{});
#else
    return BlockReduce(temp_storage).Reduce(source, cub::Min());
#endif
#elif defined(AMREX_USE_HIP)
    using BlockReduce = rocprim::block_reduce<T,BLOCKDIMX>;
    __shared__ typename BlockReduce::storage_type temp_storage;
    __syncthreads();
    BlockReduce().reduce(source, source, temp_storage, rocprim::minimum<T>());
    return source;
#else
    return blockReduceMin(source);
#endif
}

template <int BLOCKDIMX, typename T>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void deviceReduceMin_full (T * dest, T source) noexcept
{
    T aggregate = blockReduceMin<BLOCKDIMX>(source);
    if (threadIdx.x == 0) { Gpu::Atomic::Min(dest, aggregate); }
}

// Compute the block-wide reduction for thread0
template <typename T>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
T blockReduceMax (T source) noexcept
{
    return Gpu::blockReduce<Gpu::Device::warp_size>
        (source, Gpu::warpReduce<Gpu::Device::warp_size,T,amrex::Maximum<T> >(), source);
}

template <typename T>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void deviceReduceMax_full (T * dest, T source) noexcept
{
    T aggregate = blockReduceMax(source);
    if (threadIdx.x == 0) { Gpu::Atomic::Max(dest, aggregate); }
}

// Compute the block-wide reduction for thread0
template <int BLOCKDIMX, typename T>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
T blockReduceMax (T source) noexcept
{
#if defined(AMREX_USE_CUDA)
    using BlockReduce = cub::BlockReduce<T,BLOCKDIMX>;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    // __syncthreads() prior to writing to shared memory is necessary
    // if this reduction call is occurring multiple times in a kernel,
    // and since we don't know how many times the user is calling it,
    // we do it always to be safe.
    __syncthreads();
#ifdef AMREX_CUDA_CCCL_VER_GE_2_8
    return BlockReduce(temp_storage).Reduce(source, cuda::maximum<T>());
#else
    return BlockReduce(temp_storage).Reduce(source, cub::Max());
#endif
#elif defined(AMREX_USE_HIP)
    using BlockReduce = rocprim::block_reduce<T,BLOCKDIMX>;
    __shared__ typename BlockReduce::storage_type temp_storage;
    __syncthreads();
    BlockReduce().reduce(source, source, temp_storage, rocprim::maximum<T>());
    return source;
#else
    return blockReduceMax(source);
#endif
}

template <int BLOCKDIMX, typename T>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void deviceReduceMax_full (T * dest, T source) noexcept
{
    T aggregate = blockReduceMax<BLOCKDIMX>(source);
    if (threadIdx.x == 0) { Gpu::Atomic::Max(dest, aggregate); }
}

// Compute the block-wide reduction for thread0
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
int blockReduceLogicalAnd (int source) noexcept
{
    return Gpu::blockReduce<Gpu::Device::warp_size>
        (source, Gpu::warpReduce<Gpu::Device::warp_size,int,amrex::LogicalAnd<int> >(), 1);
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void deviceReduceLogicalAnd_full (int * dest, int source) noexcept
{
    int aggregate = blockReduceLogicalAnd(source);
    if (threadIdx.x == 0) { Gpu::Atomic::LogicalAnd(dest, aggregate); }
}

// Compute the block-wide reduction for thread0
template <int BLOCKDIMX>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
int blockReduceLogicalAnd (int source) noexcept
{
#if defined(AMREX_USE_CUDA)
    using BlockReduce = cub::BlockReduce<int,BLOCKDIMX>;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    // __syncthreads() prior to writing to shared memory is necessary
    // if this reduction call is occurring multiple times in a kernel,
    // and since we don't know how many times the user is calling it,
    // we do it always to be safe.
    __syncthreads();
    return BlockReduce(temp_storage).Reduce(source, amrex::LogicalAnd<int>());
#elif defined(AMREX_USE_HIP)
    using BlockReduce = rocprim::block_reduce<int,BLOCKDIMX>;
    __shared__ typename BlockReduce::storage_type temp_storage;
    __syncthreads();
    BlockReduce().reduce(source, source, temp_storage, amrex::LogicalAnd<int>());
    return source;
#else
    return blockReduceLogicalAnd(source);
#endif
}

template <int BLOCKDIMX>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void deviceReduceLogicalAnd_full (int * dest, int source) noexcept
{
    int aggregate = blockReduceLogicalAnd<BLOCKDIMX>(source);
    if (threadIdx.x == 0) { Gpu::Atomic::LogicalAnd(dest, aggregate); }
}

// Compute the block-wide reduction for thread0
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
int blockReduceLogicalOr (int source) noexcept
{
    return Gpu::blockReduce<Gpu::Device::warp_size>
        (source, Gpu::warpReduce<Gpu::Device::warp_size,int,amrex::LogicalOr<int> >(), 0);
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void deviceReduceLogicalOr_full (int * dest, int source) noexcept
{
    int aggregate = blockReduceLogicalOr(source);
    if (threadIdx.x == 0) { Gpu::Atomic::LogicalOr(dest, aggregate); }
}

// Compute the block-wide reduction for thread0
template <int BLOCKDIMX>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
int blockReduceLogicalOr (int source) noexcept
{
#if defined(AMREX_USE_CUDA)
    using BlockReduce = cub::BlockReduce<int,BLOCKDIMX>;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    // __syncthreads() prior to writing to shared memory is necessary
    // if this reduction call is occurring multiple times in a kernel,
    // and since we don't know how many times the user is calling it,
    // we do it always to be safe.
    __syncthreads();
    return BlockReduce(temp_storage).Reduce(source, amrex::LogicalOr<int>());
#elif defined(AMREX_USE_HIP)
    using BlockReduce = rocprim::block_reduce<int,BLOCKDIMX>;
    __shared__ typename BlockReduce::storage_type temp_storage;
    __syncthreads();
    BlockReduce().reduce(source, source, temp_storage, amrex::LogicalOr<int>());
    return source;
#else
    return blockReduceLogicalOr(source);
#endif
}

template <int BLOCKDIMX>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void deviceReduceLogicalOr_full (int * dest, int source) noexcept
{
    int aggregate = blockReduceLogicalOr<BLOCKDIMX>(source);
    if (threadIdx.x == 0) { Gpu::Atomic::LogicalOr(dest, aggregate); }
}

#endif

#if !defined(AMREX_USE_GPU)


template <typename T>
AMREX_FORCE_INLINE
void deviceReduceSum_full (T * dest, T source) noexcept
{
#ifdef AMREX_USE_OMP
#pragma omp atomic
#endif
    *dest += source;
}

template <typename T>
AMREX_FORCE_INLINE
void deviceReduceSum (T * dest, T source, Gpu::Handler const&) noexcept
{
    deviceReduceSum_full(dest, source);
}

template <typename T>
AMREX_FORCE_INLINE
void deviceReduceMin_full (T * dest, T source) noexcept
{
#ifdef AMREX_USE_OMP
#pragma omp critical (gpureduce_reducemin)
#endif
    *dest = std::min(*dest, source);
}

template <typename T>
AMREX_FORCE_INLINE
void deviceReduceMin (T * dest, T source, Gpu::Handler const&) noexcept
{
    deviceReduceMin_full(dest, source);
}

template <typename T>
AMREX_FORCE_INLINE
void deviceReduceMax_full (T * dest, T source) noexcept
{
#ifdef AMREX_USE_OMP
#pragma omp critical (gpureduce_reducemax)
#endif
    *dest = std::max(*dest, source);
}

template <typename T>
AMREX_FORCE_INLINE
void deviceReduceMax (T * dest, T source, Gpu::Handler const&) noexcept
{
    deviceReduceMax_full(dest, source);
}

AMREX_FORCE_INLINE
void deviceReduceLogicalAnd_full (int * dest, int source) noexcept
{
#ifdef AMREX_USE_OMP
#pragma omp critical (gpureduce_reduceand)
#endif
    *dest = (*dest) && source;
}

AMREX_FORCE_INLINE
void deviceReduceLogicalAnd (int * dest, int source, Gpu::Handler const&) noexcept
{
    deviceReduceLogicalAnd_full(dest, source);
}

AMREX_FORCE_INLINE
void deviceReduceLogicalOr_full (int * dest, int source) noexcept
{
#ifdef AMREX_USE_OMP
#pragma omp critical (gpureduce_reduceor)
#endif
    *dest = (*dest) || source;
}

AMREX_FORCE_INLINE
void deviceReduceLogicalOr (int * dest, int source, Gpu::Handler const&) noexcept
{
    deviceReduceLogicalOr_full(dest, source);
}

#endif
}

#endif
