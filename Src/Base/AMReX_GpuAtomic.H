#ifndef AMREX_GPU_ATOMIC_H_
#define AMREX_GPU_ATOMIC_H_
#include <AMReX_Config.H>

#include <AMReX_GpuQualifiers.H>
#include <AMReX_Functional.H>
#include <AMReX_INT.H>
#include <AMReX_Kokkos.H>

#include <type_traits>
#include <utility>

#if defined(AMREX_USE_CUDA) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 600)
#error "AMReX requires CUDA compute capability 6.0 or newer"
#endif

namespace amrex {

namespace Gpu::Atomic {

// For Add, Min and Max, we support int, unsigned int, long, unsigned long long, float and double.
// The result of Min and Max is unspecified if any input is NaN, and so is
// the sign of a zero result.
// For Multiply and Divide, we support generic types provided they are the same size as int or unsigned long long
// and have *= and /= operators.
// For LogicalOr and LogicalAnd, the data type is int.
// For Exch and CAS, the data type is generic.
// All these functions are non-atomic in host code!!!
// If one needs them to be atomic in host code, use HostDevice::Atomic::*.
// If we add more types for atomicAdd, we also need to update HasAtomicAdd in AMReX_TypeTraits.H.
//
// In GPU builds, the device code uses Kokkos atomics.

#ifdef AMREX_USE_GPU

/// \cond DOXYGEN_IGNORE
namespace detail {

    // Generic atomic read-modify-write through a compare-and-swap loop on
    // the bit pattern (Int) of R.
    template <typename R, typename Int, typename F>
    AMREX_GPU_DEVICE AMREX_FORCE_INLINE
    R atomic_op (R* const address, R const val, F const f) noexcept
    {
        static_assert(sizeof(R) == sizeof(Int), "sizeof R != sizeof Int");
        Int* const add_as_I = reinterpret_cast<Int*>(address);
        Int old_I = *add_as_I, assumed_I;
        do {
            assumed_I = old_I;
            R const new_R = f(*(reinterpret_cast<R const*>(&assumed_I)), val);
            old_I = Kokkos::atomic_compare_exchange(add_as_I, assumed_I,
                                                    *(reinterpret_cast<Int const*>(&new_R)));
        } while (assumed_I != old_I);
        return *(reinterpret_cast<R const*>(&old_I));
    }

    template <typename R, typename Int, typename Op, typename Cond>
    AMREX_GPU_DEVICE AMREX_FORCE_INLINE
    bool atomic_op_if (R* const address, R const val, Op&& op, Cond&& cond) noexcept
    {
        static_assert(sizeof(R) == sizeof(Int), "sizeof R != sizeof Int");
        Int* const add_as_I = reinterpret_cast<Int*>(address);
        Int old_I = *add_as_I, assumed_I;
        bool test_success;
        do {
            assumed_I = old_I;
            R const new_R = op(*(reinterpret_cast<R const*>(&assumed_I)), val);
            test_success = cond(new_R);
            if (test_success) {
                old_I = Kokkos::atomic_compare_exchange(add_as_I, assumed_I,
                                                        *(reinterpret_cast<Int const*>(&new_R)));
            }
        } while (test_success && assumed_I != old_I);
        return test_success;
    }

}
/// \endcond

#endif

////////////////////////////////////////////////////////////////////////
//  Add
////////////////////////////////////////////////////////////////////////

#ifdef AMREX_USE_GPU

// The address space parameter of the SYCL overloads is kept for
// compatibility; Kokkos atomics work on generic pointers.
#ifdef AMREX_USE_SYCL
    template<class T, sycl::access::address_space AS = sycl::access::address_space::global_space>
#else
    template<class T>
#endif
    AMREX_GPU_DEVICE AMREX_FORCE_INLINE
    T Add_device (T* const sum, T const value) noexcept
    {
        return Kokkos::atomic_fetch_add(sum, value);
    }

#if defined(AMREX_USE_HIP) && defined(__gfx90a__)
    // Hardware floating-point atomics on MI200; Kokkos uses them only with
    // -munsafe-fp-atomics.
    // https://github.com/ROCm-Developer-Tools/hipamd/blob/rocm-4.5.x/include/hip/amd_detail/amd_hip_unsafe_atomics.h
    AMREX_GPU_DEVICE AMREX_FORCE_INLINE
    float Add_device (float* const sum, float const value) noexcept
    {
        return unsafeAtomicAdd(sum, value);
    }

    AMREX_GPU_DEVICE AMREX_FORCE_INLINE
    double Add_device (double* const sum, double const value) noexcept
    {
        return unsafeAtomicAdd(sum, value);
    }
#endif

#endif

#ifdef AMREX_USE_SYCL
    template<class T, sycl::access::address_space AS = sycl::access::address_space::global_space>
#else
    template<class T>
#endif
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    T Add (T* sum, T value) noexcept
    {
#ifdef AMREX_USE_SYCL
        AMREX_IF_ON_DEVICE((return Add_device<T,AS>(sum, value);))
#else
        AMREX_IF_ON_DEVICE((return Add_device(sum, value);))
#endif
        AMREX_IF_ON_HOST((
            auto old = *sum;
            *sum += value;
            return old;
        ))
    }

////////////////////////////////////////////////////////////////////////
//  If
////////////////////////////////////////////////////////////////////////

#ifdef AMREX_USE_GPU
    template <typename T, typename Op, typename Cond>
    requires (sizeof(T) == sizeof(unsigned int))
    AMREX_GPU_DEVICE AMREX_FORCE_INLINE
    bool If_device (T* const sum, T const value, Op&& op, Cond&& cond) noexcept
    {
        return detail::atomic_op_if<T, unsigned int>(sum, value,
                   std::forward<Op>(op), std::forward<Cond>(cond));
    }

    template <typename T, typename Op, typename Cond>
    requires (sizeof(T) == sizeof(unsigned long long))
    AMREX_GPU_DEVICE AMREX_FORCE_INLINE
    bool If_device (T* const sum, T const value, Op&& op, Cond&& cond) noexcept
    {
        return detail::atomic_op_if<T, unsigned long long>(sum, value,
                   std::forward<Op>(op), std::forward<Cond>(cond));
    }
#endif

/**
*  \brief Conditionally perform an atomic operation.
*
*  Atomically updates the result at "add" with "value" using "op",
*  but only if "cond" is true.
*
* \tparam T the type pointed to by add
* \tparam Op callable that takes two T argument and combines them
* \tparam Cond callable that takes a "T" a returns whether to do the update
*
* \param add address to atomically update
* \param value value to combine
* \param op callable specifying the operation to use to combine *add and value
* \param cond callable specifying the condition to test on first.
*        The value passed in to the cond function is the would-be combined value
**/
    template<class T, class Op, class Cond>
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    bool If (T* const add, T const value, Op&& op, Cond&& cond) noexcept
    {
        AMREX_IF_ON_DEVICE((
            return If_device(add, value, std::forward<Op>(op), std::forward<Cond>(cond));
        ))
        AMREX_IF_ON_HOST((
            T old = *add;
            T const tmp = std::forward<Op>(op)(old, value);
            if (std::forward<Cond>(cond)(tmp)) {
                *add = tmp;
                return true;
            } else {
                return false;
            }
        ))
    }

////////////////////////////////////////////////////////////////////////
//  AddNoRet
////////////////////////////////////////////////////////////////////////

#ifdef AMREX_USE_SYCL
    template<class T, sycl::access::address_space AS = sycl::access::address_space::global_space>
#else
    template<class T>
#endif
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void AddNoRet (T* sum, T value) noexcept
    {
#ifdef AMREX_USE_SYCL
        AMREX_IF_ON_DEVICE((Add_device<T,AS>(sum, value);))
#else
        AMREX_IF_ON_DEVICE((Add_device(sum, value);))
#endif
        AMREX_IF_ON_HOST((*sum += value;))
    }

////////////////////////////////////////////////////////////////////////
//  Min
////////////////////////////////////////////////////////////////////////

#ifdef AMREX_USE_GPU
    template<class T>
    AMREX_GPU_DEVICE AMREX_FORCE_INLINE
    T Min_device (T* const m, T const value) noexcept
    {
        return Kokkos::atomic_fetch_min(m, value);
    }
#endif

    template<class T>
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    T Min (T* const m, T const value) noexcept
    {
        AMREX_IF_ON_DEVICE((
            return Min_device(m, value);
        ))
        AMREX_IF_ON_HOST((
            auto const old = *m;
            *m = (*m) < value ? (*m) : value;
            return old;
        ))
    }

////////////////////////////////////////////////////////////////////////
//  Max
////////////////////////////////////////////////////////////////////////

#ifdef AMREX_USE_GPU
    template<class T>
    AMREX_GPU_DEVICE AMREX_FORCE_INLINE
    T Max_device (T* const m, T const value) noexcept
    {
        return Kokkos::atomic_fetch_max(m, value);
    }
#endif

    template<class T>
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    T Max (T* const m, T const value) noexcept
    {
        AMREX_IF_ON_DEVICE((
            return Max_device(m, value);
        ))
        AMREX_IF_ON_HOST((
            auto const old = *m;
            *m = (*m) > value ? (*m) : value;
            return old;
        ))
    }

////////////////////////////////////////////////////////////////////////
//  LogicalOr
////////////////////////////////////////////////////////////////////////

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    int LogicalOr (int* const m, int const value) noexcept
    {
        AMREX_IF_ON_DEVICE((
            return Kokkos::atomic_fetch_or(m, value);
        ))
        AMREX_IF_ON_HOST((
            int const old = *m;
            *m = (*m) || value;
            return old;
        ))
    }

////////////////////////////////////////////////////////////////////////
//  LogicalAnd
////////////////////////////////////////////////////////////////////////

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    int LogicalAnd (int* const m, int const value) noexcept
    {
        AMREX_IF_ON_DEVICE((
            return Kokkos::atomic_fetch_and(m, value ? ~0x0 : 0);
        ))
        AMREX_IF_ON_HOST((
            int const old = *m;
            *m = (*m) && value;
            return old;
        ))
    }

////////////////////////////////////////////////////////////////////////
//  Exch
////////////////////////////////////////////////////////////////////////

    template <typename T>
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    T Exch (T* address, T val) noexcept
    {
        AMREX_IF_ON_DEVICE((
            return Kokkos::atomic_exchange(address, val);
        ))
        AMREX_IF_ON_HOST((
            auto const old = *address;
            *address = val;
            return old;
        ))
    }

////////////////////////////////////////////////////////////////////////
//  CAS
////////////////////////////////////////////////////////////////////////

    template <typename T>
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    T CAS (T* const address, T compare, T const val) noexcept
    {
        AMREX_IF_ON_DEVICE((
            return Kokkos::atomic_compare_exchange(address, compare, val);
        ))
        AMREX_IF_ON_HOST((
            auto const old = *address;
            *address = (old == compare ? val : old);
            return old;
        ))
    }

////////////////////////////////////////////////////////////////////////
//  Multiply
////////////////////////////////////////////////////////////////////////

#ifdef AMREX_USE_GPU

    template <typename T>
    requires (sizeof(T) == sizeof(int))
    AMREX_GPU_DEVICE AMREX_FORCE_INLINE
    T Multiply_device (T* const prod, T const value) noexcept
    {
        return detail::atomic_op<T, int>(prod,value,amrex::Multiplies<T>());
    }

    template <typename T>
    requires (sizeof(T) == sizeof(unsigned long long))
    AMREX_GPU_DEVICE AMREX_FORCE_INLINE
    T Multiply_device (T* const prod, T const value) noexcept
    {
        return detail::atomic_op<T, unsigned long long>(prod,value,amrex::Multiplies<T>());
    }

#endif

    template<class T>
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    T Multiply (T* const prod, T const value) noexcept
    {
        AMREX_IF_ON_DEVICE((
            return Multiply_device(prod, value);
        ))
        AMREX_IF_ON_HOST((
            auto const old = *prod;
            *prod *= value;
            return old;
        ))
    }

////////////////////////////////////////////////////////////////////////
//  Divide
////////////////////////////////////////////////////////////////////////

#ifdef AMREX_USE_GPU

    template <typename T>
    requires (sizeof(T) == sizeof(int))
    AMREX_GPU_DEVICE AMREX_FORCE_INLINE
    T Divide_device (T* const quot, T const value) noexcept
    {
        return detail::atomic_op<T, int>(quot,value,amrex::Divides<T>());
    }

    template <typename T>
    requires (sizeof(T) == sizeof(unsigned long long))
    AMREX_GPU_DEVICE AMREX_FORCE_INLINE
    T Divide_device (T* const quot, T const value) noexcept
    {
        return detail::atomic_op<T, unsigned long long>(quot,value,amrex::Divides<T>());
    }

#endif

    template<class T>
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    T Divide (T* const quot, T const value) noexcept
    {
        AMREX_IF_ON_DEVICE((
            return Divide_device(quot, value);
        ))
        AMREX_IF_ON_HOST((
            auto const old = *quot;
            *quot /= value;
            return old;
        ))
    }
}

namespace HostDevice::Atomic {

    template <class T>
    AMREX_FORCE_INLINE
    void Add_Host (T* const sum, T const value) noexcept
    {
#ifdef AMREX_USE_OMP
#pragma omp atomic update
#endif
        *sum += value;
    }

    template <class T>
    AMREX_FORCE_INLINE
    T FetchAdd_Host (T* const sum, T const value) noexcept
    {
        T old;
#ifdef AMREX_USE_OMP
#pragma omp atomic capture
#endif
        {
            old = *sum;
            *sum += value;
        }
        return old;
    }

    template <class T>
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void Add (T* const sum, T const value) noexcept
    {
        AMREX_IF_ON_DEVICE((Gpu::Atomic::AddNoRet(sum,value);))
        AMREX_IF_ON_HOST((Add_Host(sum,value);))
    }

    template <class T>
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    T FetchAdd (T* const sum, T const value) noexcept
    {
        AMREX_IF_ON_DEVICE((return Gpu::Atomic::Add(sum,value);))
        AMREX_IF_ON_HOST((return FetchAdd_Host(sum,value);))
    }

}

#ifdef AMREX_USE_GPU
// functors
namespace Gpu {
    template <typename T>
    struct AtomicAdd
    {
        AMREX_GPU_DEVICE void operator() (T* const dest, T const source) noexcept {
            Gpu::Atomic::AddNoRet(dest, source);
        }
    };

    template <typename T>
    struct AtomicMin
    {
        AMREX_GPU_DEVICE void operator() (T* const dest, T const source) noexcept {
            Gpu::Atomic::Min(dest, source);
        }
    };

    template <typename T>
    struct AtomicMax
    {
        AMREX_GPU_DEVICE void operator() (T* const dest, T const source) noexcept {
            Gpu::Atomic::Max(dest, source);
        }
    };

    template <typename T>
    struct AtomicLogicalAnd
    {
        AMREX_GPU_DEVICE void operator() (T* const dest, T const source) noexcept {
            Gpu::Atomic::LogicalAnd(dest, source);
        }
    };

    template <typename T>
    struct AtomicLogicalOr
    {
        AMREX_GPU_DEVICE void operator() (T* const dest, T const source) noexcept {
            Gpu::Atomic::LogicalOr(dest, source);
        }
    };
}
#endif

}
#endif
