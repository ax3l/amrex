#ifndef AMREX_GPU_LAUNCH_FUNCTS_G_H_
#define AMREX_GPU_LAUNCH_FUNCTS_G_H_
#include <AMReX_Config.H>

namespace amrex {

/// \cond DOXYGEN_IGNORE
namespace detail {

    // call_f_scalar_handler

    template <typename F, typename N>
    AMREX_GPU_DEVICE AMREX_FORCE_INLINE
    auto call_f_scalar_handler (F const& f, N i, Gpu::Handler const&)
        noexcept -> decltype(f(0))
    {
        return f(i);
    }

    template <typename F, typename N>
    AMREX_GPU_DEVICE AMREX_FORCE_INLINE
    auto call_f_scalar_handler (F const& f, N i, Gpu::Handler const& handler)
        noexcept -> decltype(f(0,Gpu::Handler{}))
    {
        return f(i, handler);
    }

    // call_f_intvect_inner

    template <typename F, std::size_t...Ns, class...Args>
    AMREX_GPU_DEVICE AMREX_FORCE_INLINE
    auto call_f_intvect_inner (std::index_sequence<Ns...>, F const& f, IntVectND<1> iv, Args...args)
        noexcept -> decltype(f(0, 0, 0, args...))
    {
        return f(iv[0], 0, 0, args...);
    }

    template <typename F, std::size_t...Ns, class...Args>
    AMREX_GPU_DEVICE AMREX_FORCE_INLINE
    auto call_f_intvect_inner (std::index_sequence<Ns...>, F const& f, IntVectND<2> iv, Args...args)
        noexcept -> decltype(f(0, 0, 0, args...))
    {
        return f(iv[0], iv[1], 0, args...);
    }

    template <typename F, int dim, std::size_t...Ns, class...Args>
    AMREX_GPU_DEVICE AMREX_FORCE_INLINE
    auto call_f_intvect_inner (std::index_sequence<Ns...>, F const& f, IntVectND<dim> iv, Args...args)
        noexcept -> decltype(f(iv, args...))
    {
        return f(iv, args...);
    }

    template <typename F, int dim, std::size_t...Ns, class...Args>
    AMREX_GPU_DEVICE AMREX_FORCE_INLINE
    auto call_f_intvect_inner (std::index_sequence<Ns...>, F const& f, IntVectND<dim> iv, Args...args)
        noexcept -> decltype(f(iv[Ns]..., args...))
    {
        return f(iv[Ns]..., args...);
    }

    // call_f_intvect

    template <typename F, int dim>
    AMREX_GPU_DEVICE AMREX_FORCE_INLINE
    auto call_f_intvect (F const& f, IntVectND<dim> iv)
        noexcept -> decltype(call_f_intvect_inner(std::make_index_sequence<dim>(), f, iv))
    {
        return call_f_intvect_inner(std::make_index_sequence<dim>(), f, iv);
    }

    // call_f_intvect_engine

    template <typename F, int dim>
    AMREX_GPU_DEVICE AMREX_FORCE_INLINE
    auto call_f_intvect_engine (F const& f, IntVectND<dim> iv, RandomEngine engine)
        noexcept -> decltype(call_f_intvect_inner(std::make_index_sequence<dim>(), f, iv, engine))
    {
        return call_f_intvect_inner(std::make_index_sequence<dim>(), f, iv, engine);
    }

    // call_f_intvect_handler

    template <typename F, int dim>
    AMREX_GPU_DEVICE AMREX_FORCE_INLINE
    auto call_f_intvect_handler (F const& f, IntVectND<dim> iv, Gpu::Handler const&)
        noexcept -> decltype(call_f_intvect_inner(std::make_index_sequence<dim>(), f, iv))
    {
        return call_f_intvect_inner(std::make_index_sequence<dim>(), f, iv);
    }

    template <typename F, int dim>
    AMREX_GPU_DEVICE AMREX_FORCE_INLINE
    auto call_f_intvect_handler (F const& f, IntVectND<dim> iv, Gpu::Handler const& handler)
        noexcept -> decltype(call_f_intvect_inner(std::make_index_sequence<dim>(), f, iv, Gpu::Handler{}))
    {
        return call_f_intvect_inner(std::make_index_sequence<dim>(), f, iv, handler);
    }

    // call_f_intvect_ncomp

    template <typename F, typename T, int dim>
    AMREX_GPU_DEVICE AMREX_FORCE_INLINE
    auto call_f_intvect_ncomp (F const& f, IntVectND<dim> iv, T ncomp)
        noexcept -> decltype(call_f_intvect_inner(std::make_index_sequence<dim>(), f, iv, 0))
    {
        for (T n = 0; n < ncomp; ++n) {
            call_f_intvect_inner(std::make_index_sequence<dim>(), f, iv, n);
        }
    }

    // call_f_intvect_ncomp_engine

    template <typename F, typename T, int dim>
    AMREX_GPU_DEVICE AMREX_FORCE_INLINE
    auto call_f_intvect_ncomp_engine (F const& f, IntVectND<dim> iv, T ncomp, RandomEngine engine)
        noexcept -> decltype(call_f_intvect_inner(std::make_index_sequence<dim>(), f, iv, 0, engine))
    {
        for (T n = 0; n < ncomp; ++n) {
            call_f_intvect_inner(std::make_index_sequence<dim>(), f, iv, n, engine);
        }
    }

    // call_f_intvect_ncomp_handler

    template <typename F, typename T, int dim>
    AMREX_GPU_DEVICE AMREX_FORCE_INLINE
    auto call_f_intvect_ncomp_handler (F const& f, IntVectND<dim> iv, T ncomp, Gpu::Handler const&)
        noexcept -> decltype(call_f_intvect_inner(std::make_index_sequence<dim>(), f, iv, 0))
    {
        for (T n = 0; n < ncomp; ++n) {
            call_f_intvect_inner(std::make_index_sequence<dim>(), f, iv, n);
        }
    }

    template <typename F, typename T, int dim>
    AMREX_GPU_DEVICE AMREX_FORCE_INLINE
    auto call_f_intvect_ncomp_handler (F const& f, IntVectND<dim> iv, T ncomp, Gpu::Handler const& handler)
        noexcept -> decltype(call_f_intvect_inner(std::make_index_sequence<dim>(), f, iv, 0, Gpu::Handler{}))
    {
        for (T n = 0; n < ncomp; ++n) {
            call_f_intvect_inner(std::make_index_sequence<dim>(), f, iv, n, handler);
        }
    }

}
/// \endcond

#ifdef AMREX_USE_SYCL

template <typename L>
void single_task (gpuStream_t stream, L const& f)
{
    detail::SyclKernelDevPtr<L> skdp(f, stream);
    L const* pf = skdp.template get<0>();
    amrex::ignore_unused(pf);

    auto& q = *(stream.queue);
    try {
        q.submit([&] (sycl::handler& h) {
            if constexpr (detail::is_big_kernel<L>()) {
                h.single_task([=] () { (*pf)(); });
            } else {
                h.single_task(f);
            }
        });
    } catch (sycl::exception const& ex) {
        amrex::Abort(std::string("single_task: ")+ex.what()+"!!!!!");
    }
}

template<typename L>
void launch (int nblocks, int nthreads_per_block, std::size_t shared_mem_bytes,
             gpuStream_t stream, L const& f)
{
    detail::SyclKernelDevPtr<L> skdp(f, stream);
    L const* pf = skdp.template get<0>();
    amrex::ignore_unused(pf);

    const auto nthreads_total = std::size_t(nthreads_per_block) * nblocks;
    const std::size_t shared_mem_numull = (shared_mem_bytes+sizeof(unsigned long long)-1)
        / sizeof(unsigned long long);
    auto& q = *(stream.queue);
    try {
        q.submit([&] (sycl::handler& h) {
            sycl::local_accessor<unsigned long long>
                shared_data(sycl::range<1>(shared_mem_numull), h);
            h.parallel_for(sycl::nd_range<1>(sycl::range<1>(nthreads_total),
                                             sycl::range<1>(nthreads_per_block)),
            [=] (sycl::nd_item<1> item)
            [[sycl::reqd_sub_group_size(Gpu::Device::warp_size)]]
            {
                if constexpr (detail::is_big_kernel<L>()) {
                    (*pf)(Gpu::Handler{&item,shared_data.get_multi_ptr<sycl::access::decorated::yes>().get()});
                } else {
                    f(Gpu::Handler{&item,shared_data.get_multi_ptr<sycl::access::decorated::yes>().get()});
                }
            });
        });
    } catch (sycl::exception const& ex) {
        amrex::Abort(std::string("launch: ")+ex.what()+"!!!!!");
    }
}

template<typename L>
void launch (int nblocks, int nthreads_per_block, gpuStream_t stream, L const& f)
{
    detail::SyclKernelDevPtr<L> skdp(f, stream);
    L const* pf = skdp.template get<0>();
    amrex::ignore_unused(pf);

    const auto nthreads_total = std::size_t(nthreads_per_block) * nblocks;
    auto& q = *(stream.queue);
    try {
        q.submit([&] (sycl::handler& h) {
            h.parallel_for(sycl::nd_range<1>(sycl::range<1>(nthreads_total),
                                             sycl::range<1>(nthreads_per_block)),
            [=] (sycl::nd_item<1> item)
            [[sycl::reqd_sub_group_size(Gpu::Device::warp_size)]]
            {
                if constexpr (detail::is_big_kernel<L>()) {
                    (*pf)(item);
                } else {
                    f(item);
                }
            });
        });
    } catch (sycl::exception const& ex) {
        amrex::Abort(std::string("launch: ")+ex.what()+"!!!!!");
    }
}

template <int MT, typename L>
void launch (int nblocks, std::size_t shared_mem_bytes, gpuStream_t stream,
             L const& f)
{
    detail::SyclKernelDevPtr<L> skdp(f, stream);
    L const* pf = skdp.template get<0>();
    amrex::ignore_unused(pf);

    const auto nthreads_total = MT * std::size_t(nblocks);
    const std::size_t shared_mem_numull = (shared_mem_bytes+sizeof(unsigned long long)-1)
        / sizeof(unsigned long long);
    auto& q = *(stream.queue);
    try {
        q.submit([&] (sycl::handler& h) {
            sycl::local_accessor<unsigned long long>
                shared_data(sycl::range<1>(shared_mem_numull), h);
            h.parallel_for(sycl::nd_range<1>(sycl::range<1>(nthreads_total),
                                             sycl::range<1>(MT)),
            [=] (sycl::nd_item<1> item)
            [[sycl::reqd_work_group_size(MT)]]
            [[sycl::reqd_sub_group_size(Gpu::Device::warp_size)]]
            {
                if constexpr (detail::is_big_kernel<L>()) {
                    (*pf)(Gpu::Handler{&item,shared_data.get_multi_ptr<sycl::access::decorated::yes>().get()});
                } else {
                    f(Gpu::Handler{&item,shared_data.get_multi_ptr<sycl::access::decorated::yes>().get()});
                }
            });
        });
    } catch (sycl::exception const& ex) {
        amrex::Abort(std::string("launch: ")+ex.what()+"!!!!!");
    }
}

template <int MT, typename L>
void launch (int nblocks, gpuStream_t stream, L const& f)
{
    detail::SyclKernelDevPtr<L> skdp(f, stream);
    L const* pf = skdp.template get<0>();
    amrex::ignore_unused(pf);

    const auto nthreads_total = MT * std::size_t(nblocks);
    auto& q = *(stream.queue);
    try {
        q.submit([&] (sycl::handler& h) {
            h.parallel_for(sycl::nd_range<1>(sycl::range<1>(nthreads_total),
                                             sycl::range<1>(MT)),
            [=] (sycl::nd_item<1> item)
            [[sycl::reqd_work_group_size(MT)]]
            [[sycl::reqd_sub_group_size(Gpu::Device::warp_size)]]
            {
                if constexpr (detail::is_big_kernel<L>()) {
                    (*pf)(item);
                } else {
                    f(item);
                }
            });
        });
    } catch (sycl::exception const& ex) {
        amrex::Abort(std::string("launch: ")+ex.what()+"!!!!!");
    }
}

#else
// CUDA or HIP

template <typename L>
void single_task (gpuStream_t stream, L const& f)
{
    AMREX_LAUNCH_KERNEL(Gpu::Device::warp_size, 1, 1, 0, stream, f);
    AMREX_GPU_ERROR_CHECK();
}

template <int MT, typename L>
void launch (int nblocks, std::size_t shared_mem_bytes, gpuStream_t stream,
             L const& f)
{
    AMREX_LAUNCH_KERNEL(MT, nblocks, MT, shared_mem_bytes, stream, f);
    AMREX_GPU_ERROR_CHECK();
}

template <int MT, typename L>
void launch (int nblocks, gpuStream_t stream, L const& f)
{
    AMREX_LAUNCH_KERNEL(MT, nblocks, MT, 0, stream, f);
    AMREX_GPU_ERROR_CHECK();
}

template<typename L>
void launch (int nblocks, int nthreads_per_block, std::size_t shared_mem_bytes,
             gpuStream_t stream, L const& f)
{
    AMREX_LAUNCH_KERNEL_NOBOUND(nblocks, nthreads_per_block, shared_mem_bytes, stream, f);
    AMREX_GPU_ERROR_CHECK();
}

template<typename L>
void launch (int nblocks, int nthreads_per_block, gpuStream_t stream, L&& f) noexcept
{
    launch(nblocks, nthreads_per_block, 0, stream, std::forward<L>(f));
}

#endif

} // namespace amrex

#include <AMReX_GpuLaunchFunctsKokkos.H>

namespace amrex {

template <typename L>
void single_task (L&& f) noexcept
{
    single_task(Gpu::gpuStream(), std::forward<L>(f));
}

template<typename T, typename L>
void launch (T const& n, L&& f) noexcept
{
    launch<AMREX_GPU_MAX_THREADS>(n, std::forward<L>(f));
}

/**
 * \ingroup amrex_execution
 * \brief Performance-portable kernel launch function
 */
template <std::integral T, typename L>
requires (MaybeDeviceRunnable<L>::value)
void
ParallelFor (Gpu::KernelInfo const& info, T n, L&& f) noexcept
{
    ParallelFor<AMREX_GPU_MAX_THREADS>(info, n, std::forward<L>(f));
}

/**
 * \ingroup amrex_execution
 * \brief Performance-portable kernel launch function
 */
template <typename L, int dim>
requires (MaybeDeviceRunnable<L>::value)
void
ParallelFor (Gpu::KernelInfo const& info, BoxND<dim> const& box, L&& f) noexcept
{
    ParallelFor<AMREX_GPU_MAX_THREADS>(info, box, std::forward<L>(f));
}

/**
 * \ingroup amrex_execution
 * \brief Performance-portable kernel launch function
 */
template <std::integral T, typename L, int dim>
requires (MaybeDeviceRunnable<L>::value)
void
ParallelFor (Gpu::KernelInfo const& info, BoxND<dim> const& box, T ncomp, L&& f) noexcept
{
    ParallelFor<AMREX_GPU_MAX_THREADS>(info, box, ncomp, std::forward<L>(f));
}

/**
 * \ingroup amrex_execution
 * \brief Performance-portable kernel launch function
 */
template <typename L1, typename L2, int dim>
requires (MaybeDeviceRunnable<L1>::value && MaybeDeviceRunnable<L2>::value)
void
ParallelFor (Gpu::KernelInfo const& info,
             BoxND<dim> const& box1, BoxND<dim> const& box2, L1&& f1, L2&& f2) noexcept
{
    ParallelFor<AMREX_GPU_MAX_THREADS>(info, box1, box2, std::forward<L1>(f1),
                                       std::forward<L2>(f2));
}

/**
 * \ingroup amrex_execution
 * \brief Performance-portable kernel launch function
 */
template <typename L1, typename L2, typename L3, int dim>
requires (MaybeDeviceRunnable<L1>::value && MaybeDeviceRunnable<L2>::value &&
          MaybeDeviceRunnable<L3>::value)
void
ParallelFor (Gpu::KernelInfo const& info,
             BoxND<dim> const& box1, BoxND<dim> const& box2, BoxND<dim> const& box3,
             L1&& f1, L2&& f2, L3&& f3) noexcept
{
    ParallelFor<AMREX_GPU_MAX_THREADS>(info, box1, box2, box3, std::forward<L1>(f1),
                                       std::forward<L2>(f2), std::forward<L3>(f3));
}

/**
 * \ingroup amrex_execution
 * \brief Performance-portable kernel launch function
 */
template <std::integral T1, std::integral T2, typename L1, typename L2, int dim>
requires (MaybeDeviceRunnable<L1>::value && MaybeDeviceRunnable<L2>::value)
void
ParallelFor (Gpu::KernelInfo const& info,
             BoxND<dim> const& box1, T1 ncomp1, L1&& f1,
             BoxND<dim> const& box2, T2 ncomp2, L2&& f2) noexcept
{
    ParallelFor<AMREX_GPU_MAX_THREADS>(info, box1, ncomp1, std::forward<L1>(f1),
                                             box2, ncomp2, std::forward<L2>(f2));
}

/**
 * \ingroup amrex_execution
 * \brief Performance-portable kernel launch function
 */
template <std::integral T1, std::integral T2, std::integral T3, typename L1, typename L2, typename L3, int dim>
requires (MaybeDeviceRunnable<L1>::value && MaybeDeviceRunnable<L2>::value &&
          MaybeDeviceRunnable<L3>::value)
void
ParallelFor (Gpu::KernelInfo const& info,
             BoxND<dim> const& box1, T1 ncomp1, L1&& f1,
             BoxND<dim> const& box2, T2 ncomp2, L2&& f2,
             BoxND<dim> const& box3, T3 ncomp3, L3&& f3) noexcept
{
    ParallelFor<AMREX_GPU_MAX_THREADS>(info, box1, ncomp1, std::forward<L1>(f1),
                                             box2, ncomp2, std::forward<L2>(f2),
                                             box3, ncomp3, std::forward<L3>(f3));
}

template <std::integral T, typename L>
void For (Gpu::KernelInfo const& info, T n, L&& f) noexcept
{
    ParallelFor<AMREX_GPU_MAX_THREADS>(info, n,std::forward<L>(f));
}

template <int MT, std::integral T, typename L>
void For (Gpu::KernelInfo const& info, T n, L&& f) noexcept
{
    ParallelFor<MT>(info, n,std::forward<L>(f));
}

template <typename L, int dim>
void For (Gpu::KernelInfo const& info, BoxND<dim> const& box, L&& f) noexcept
{
    ParallelFor<AMREX_GPU_MAX_THREADS>(info, box,std::forward<L>(f));
}

template <int MT, typename L, int dim>
void For (Gpu::KernelInfo const& info, BoxND<dim> const& box, L&& f) noexcept
{
    ParallelFor<MT>(info, box,std::forward<L>(f));
}

template <std::integral T, typename L, int dim>
void For (Gpu::KernelInfo const& info, BoxND<dim> const& box, T ncomp, L&& f) noexcept
{
    ParallelFor<AMREX_GPU_MAX_THREADS>(info,box,ncomp,std::forward<L>(f));
}

template <int MT, std::integral T, typename L, int dim>
void For (Gpu::KernelInfo const& info, BoxND<dim> const& box, T ncomp, L&& f) noexcept
{
    ParallelFor<MT>(info,box,ncomp,std::forward<L>(f));
}

template <typename L1, typename L2, int dim>
void For (Gpu::KernelInfo const& info,
          BoxND<dim> const& box1, BoxND<dim> const& box2, L1&& f1, L2&& f2) noexcept
{
    ParallelFor<AMREX_GPU_MAX_THREADS>(info,box1,box2,std::forward<L1>(f1),std::forward<L2>(f2));
}

template <int MT, typename L1, typename L2, int dim>
void For (Gpu::KernelInfo const& info,
          BoxND<dim> const& box1, BoxND<dim> const& box2, L1&& f1, L2&& f2) noexcept
{
    ParallelFor<MT>(info,box1,box2,std::forward<L1>(f1),std::forward<L2>(f2));
}

template <typename L1, typename L2, typename L3, int dim>
void For (Gpu::KernelInfo const& info,
          BoxND<dim> const& box1, BoxND<dim> const& box2, BoxND<dim> const& box3,
          L1&& f1, L2&& f2, L3&& f3) noexcept
{
    ParallelFor<AMREX_GPU_MAX_THREADS>(info,box1,box2,box3,std::forward<L1>(f1),std::forward<L2>(f2),std::forward<L3>(f3));
}

template <int MT, typename L1, typename L2, typename L3, int dim>
void For (Gpu::KernelInfo const& info,
          BoxND<dim> const& box1, BoxND<dim> const& box2, BoxND<dim> const& box3,
          L1&& f1, L2&& f2, L3&& f3) noexcept
{
    ParallelFor<MT>(info,box1,box2,box3,std::forward<L1>(f1),std::forward<L2>(f2),std::forward<L3>(f3));
}

template <std::integral T1, std::integral T2, typename L1, typename L2, int dim>
void For (Gpu::KernelInfo const& info,
          BoxND<dim> const& box1, T1 ncomp1, L1&& f1,
          BoxND<dim> const& box2, T2 ncomp2, L2&& f2) noexcept
{
    ParallelFor<AMREX_GPU_MAX_THREADS>(info,box1,ncomp1,std::forward<L1>(f1),box2,ncomp2,std::forward<L2>(f2));
}

template <int MT, std::integral T1, std::integral T2, typename L1, typename L2, int dim>
void For (Gpu::KernelInfo const& info,
          BoxND<dim> const& box1, T1 ncomp1, L1&& f1,
          BoxND<dim> const& box2, T2 ncomp2, L2&& f2) noexcept
{
    ParallelFor<MT>(info,box1,ncomp1,std::forward<L1>(f1),box2,ncomp2,std::forward<L2>(f2));
}

template <std::integral T1, std::integral T2, std::integral T3, typename L1, typename L2, typename L3, int dim>
void For (Gpu::KernelInfo const& info,
          BoxND<dim> const& box1, T1 ncomp1, L1&& f1,
          BoxND<dim> const& box2, T2 ncomp2, L2&& f2,
          BoxND<dim> const& box3, T3 ncomp3, L3&& f3) noexcept
{
    ParallelFor<AMREX_GPU_MAX_THREADS>(info,
                box1,ncomp1,std::forward<L1>(f1),
                box2,ncomp2,std::forward<L2>(f2),
                box3,ncomp3,std::forward<L3>(f3));
}

template <int MT, std::integral T1, std::integral T2, std::integral T3, typename L1, typename L2, typename L3, int dim>
void For (Gpu::KernelInfo const& info,
          BoxND<dim> const& box1, T1 ncomp1, L1&& f1,
          BoxND<dim> const& box2, T2 ncomp2, L2&& f2,
          BoxND<dim> const& box3, T3 ncomp3, L3&& f3) noexcept
{
    ParallelFor<MT>(info,
                box1,ncomp1,std::forward<L1>(f1),
                box2,ncomp2,std::forward<L2>(f2),
                box3,ncomp3,std::forward<L3>(f3));
}

/**
 * \ingroup amrex_execution
 * \brief Performance-portable kernel launch function
 */
template <std::integral T, typename L>
void ParallelFor (T n, L&& f) noexcept
{
    ParallelFor<AMREX_GPU_MAX_THREADS>(Gpu::KernelInfo{}, n, std::forward<L>(f));
}

/**
 * \ingroup amrex_execution
 * \brief Performance-portable kernel launch function
 */
template <int MT, std::integral T, typename L>
void ParallelFor (T n, L&& f) noexcept
{
    ParallelFor<MT>(Gpu::KernelInfo{}, n, std::forward<L>(f));
}

/**
 * \ingroup amrex_execution
 * \brief Performance-portable kernel launch function
 */
template <typename L, int dim>
void ParallelFor (BoxND<dim> const& box, L&& f) noexcept
{
    ParallelFor<AMREX_GPU_MAX_THREADS>(Gpu::KernelInfo{}, box, std::forward<L>(f));
}

/**
 * \ingroup amrex_execution
 * \brief Performance-portable kernel launch function
 */
template <int MT, typename L, int dim>
void ParallelFor (BoxND<dim> const& box, L&& f) noexcept
{
    ParallelFor<MT>(Gpu::KernelInfo{}, box, std::forward<L>(f));
}

/**
 * \ingroup amrex_execution
 * \brief Performance-portable kernel launch function
 */
template <std::integral T, typename L, int dim>
void ParallelFor (BoxND<dim> const& box, T ncomp, L&& f) noexcept
{
    ParallelFor<AMREX_GPU_MAX_THREADS>(Gpu::KernelInfo{},box,ncomp,std::forward<L>(f));
}

/**
 * \ingroup amrex_execution
 * \brief Performance-portable kernel launch function
 */
template <int MT, std::integral T, typename L, int dim>
void ParallelFor (BoxND<dim> const& box, T ncomp, L&& f) noexcept
{
    ParallelFor<MT>(Gpu::KernelInfo{},box,ncomp,std::forward<L>(f));
}

/**
 * \ingroup amrex_execution
 * \brief Performance-portable kernel launch function
 */
template <typename L1, typename L2, int dim>
void ParallelFor (BoxND<dim> const& box1, BoxND<dim> const& box2, L1&& f1, L2&& f2) noexcept
{
    ParallelFor<AMREX_GPU_MAX_THREADS>(Gpu::KernelInfo{},box1,box2,std::forward<L1>(f1),std::forward<L2>(f2));
}

/**
 * \ingroup amrex_execution
 * \brief Performance-portable kernel launch function
 */
template <int MT, typename L1, typename L2, int dim>
void ParallelFor (BoxND<dim> const& box1, BoxND<dim> const& box2, L1&& f1, L2&& f2) noexcept
{
    ParallelFor<MT>(Gpu::KernelInfo{},box1,box2,std::forward<L1>(f1),std::forward<L2>(f2));
}

/**
 * \ingroup amrex_execution
 * \brief Performance-portable kernel launch function
 */
template <typename L1, typename L2, typename L3, int dim>
void ParallelFor (BoxND<dim> const& box1, BoxND<dim> const& box2, BoxND<dim> const& box3,
                  L1&& f1, L2&& f2, L3&& f3) noexcept
{
    ParallelFor<AMREX_GPU_MAX_THREADS>(Gpu::KernelInfo{},box1,box2,box3,std::forward<L1>(f1),std::forward<L2>(f2),std::forward<L3>(f3));
}

/**
 * \ingroup amrex_execution
 * \brief Performance-portable kernel launch function
 */
template <int MT, typename L1, typename L2, typename L3, int dim>
void ParallelFor (BoxND<dim> const& box1, BoxND<dim> const& box2, BoxND<dim> const& box3,
                  L1&& f1, L2&& f2, L3&& f3) noexcept
{
    ParallelFor<MT>(Gpu::KernelInfo{},box1,box2,box3,std::forward<L1>(f1),std::forward<L2>(f2),std::forward<L3>(f3));
}

/**
 * \ingroup amrex_execution
 * \brief Performance-portable kernel launch function
 */
template <std::integral T1, std::integral T2, typename L1, typename L2, int dim>
void ParallelFor (BoxND<dim> const& box1, T1 ncomp1, L1&& f1,
                  BoxND<dim> const& box2, T2 ncomp2, L2&& f2) noexcept
{
    ParallelFor<AMREX_GPU_MAX_THREADS>(Gpu::KernelInfo{},box1,ncomp1,std::forward<L1>(f1),box2,ncomp2,std::forward<L2>(f2));
}

/**
 * \ingroup amrex_execution
 * \brief Performance-portable kernel launch function
 */
template <int MT, std::integral T1, std::integral T2, typename L1, typename L2, int dim>
void ParallelFor (BoxND<dim> const& box1, T1 ncomp1, L1&& f1,
                  BoxND<dim> const& box2, T2 ncomp2, L2&& f2) noexcept
{
    ParallelFor<MT>(Gpu::KernelInfo{},box1,ncomp1,std::forward<L1>(f1),box2,ncomp2,std::forward<L2>(f2));
}

/**
 * \ingroup amrex_execution
 * \brief Performance-portable kernel launch function
 */
template <std::integral T1, std::integral T2, std::integral T3, typename L1, typename L2, typename L3, int dim>
void ParallelFor (BoxND<dim> const& box1, T1 ncomp1, L1&& f1,
                  BoxND<dim> const& box2, T2 ncomp2, L2&& f2,
                  BoxND<dim> const& box3, T3 ncomp3, L3&& f3) noexcept
{
    ParallelFor<AMREX_GPU_MAX_THREADS>(Gpu::KernelInfo{},
                box1,ncomp1,std::forward<L1>(f1),
                box2,ncomp2,std::forward<L2>(f2),
                box3,ncomp3,std::forward<L3>(f3));
}

/**
 * \ingroup amrex_execution
 * \brief Performance-portable kernel launch function
 */
template <int MT, std::integral T1, std::integral T2, std::integral T3, typename L1, typename L2, typename L3, int dim>
void ParallelFor (BoxND<dim> const& box1, T1 ncomp1, L1&& f1,
                  BoxND<dim> const& box2, T2 ncomp2, L2&& f2,
                  BoxND<dim> const& box3, T3 ncomp3, L3&& f3) noexcept
{
    ParallelFor<MT>(Gpu::KernelInfo{},
                box1,ncomp1,std::forward<L1>(f1),
                box2,ncomp2,std::forward<L2>(f2),
                box3,ncomp3,std::forward<L3>(f3));
}

template <std::integral T, typename L>
void For (T n, L&& f) noexcept
{
    ParallelFor<AMREX_GPU_MAX_THREADS>(Gpu::KernelInfo{}, n,std::forward<L>(f));
}

template <int MT, std::integral T, typename L>
void For (T n, L&& f) noexcept
{
    ParallelFor<MT>(Gpu::KernelInfo{}, n,std::forward<L>(f));
}

template <typename L, int dim>
void For (BoxND<dim> const& box, L&& f) noexcept
{
    ParallelFor<AMREX_GPU_MAX_THREADS>(Gpu::KernelInfo{}, box,std::forward<L>(f));
}

template <int MT, typename L, int dim>
void For (BoxND<dim> const& box, L&& f) noexcept
{
    ParallelFor<MT>(Gpu::KernelInfo{}, box,std::forward<L>(f));
}

template <std::integral T, typename L, int dim>
void For (BoxND<dim> const& box, T ncomp, L&& f) noexcept
{
    ParallelFor<AMREX_GPU_MAX_THREADS>(Gpu::KernelInfo{},box,ncomp,std::forward<L>(f));
}

template <int MT, std::integral T, typename L, int dim>
void For (BoxND<dim> const& box, T ncomp, L&& f) noexcept
{
    ParallelFor<MT>(Gpu::KernelInfo{},box,ncomp,std::forward<L>(f));
}

template <typename L1, typename L2, int dim>
void For (BoxND<dim> const& box1, BoxND<dim> const& box2, L1&& f1, L2&& f2) noexcept
{
    ParallelFor<AMREX_GPU_MAX_THREADS>(Gpu::KernelInfo{},box1,box2,std::forward<L1>(f1),std::forward<L2>(f2));
}

template <int MT, typename L1, typename L2, int dim>
void For (BoxND<dim> const& box1, BoxND<dim> const& box2, L1&& f1, L2&& f2) noexcept
{
    ParallelFor<MT>(Gpu::KernelInfo{},box1,box2,std::forward<L1>(f1),std::forward<L2>(f2));
}

template <typename L1, typename L2, typename L3, int dim>
void For (BoxND<dim> const& box1, BoxND<dim> const& box2, BoxND<dim> const& box3,
          L1&& f1, L2&& f2, L3&& f3) noexcept
{
    ParallelFor<AMREX_GPU_MAX_THREADS>(Gpu::KernelInfo{},box1,box2,box3,std::forward<L1>(f1),std::forward<L2>(f2),std::forward<L3>(f3));
}

template <int MT, typename L1, typename L2, typename L3, int dim>
void For (BoxND<dim> const& box1, BoxND<dim> const& box2, BoxND<dim> const& box3,
          L1&& f1, L2&& f2, L3&& f3) noexcept
{
    ParallelFor<MT>(Gpu::KernelInfo{},box1,box2,box3,std::forward<L1>(f1),std::forward<L2>(f2),std::forward<L3>(f3));
}

template <std::integral T1, std::integral T2, typename L1, typename L2, int dim>
void For (BoxND<dim> const& box1, T1 ncomp1, L1&& f1,
          BoxND<dim> const& box2, T2 ncomp2, L2&& f2) noexcept
{
    ParallelFor<AMREX_GPU_MAX_THREADS>(Gpu::KernelInfo{},box1,ncomp1,std::forward<L1>(f1),box2,ncomp2,std::forward<L2>(f2));
}

template <int MT, std::integral T1, std::integral T2, typename L1, typename L2, int dim>
void For (BoxND<dim> const& box1, T1 ncomp1, L1&& f1,
          BoxND<dim> const& box2, T2 ncomp2, L2&& f2) noexcept
{
    ParallelFor<MT>(Gpu::KernelInfo{},box1,ncomp1,std::forward<L1>(f1),box2,ncomp2,std::forward<L2>(f2));
}

template <std::integral T1, std::integral T2, std::integral T3, typename L1, typename L2, typename L3, int dim>
void For (BoxND<dim> const& box1, T1 ncomp1, L1&& f1,
          BoxND<dim> const& box2, T2 ncomp2, L2&& f2,
          BoxND<dim> const& box3, T3 ncomp3, L3&& f3) noexcept
{
    ParallelFor<AMREX_GPU_MAX_THREADS>(Gpu::KernelInfo{},
                box1,ncomp1,std::forward<L1>(f1),
                box2,ncomp2,std::forward<L2>(f2),
                box3,ncomp3,std::forward<L3>(f3));
}

template <int MT, std::integral T1, std::integral T2, std::integral T3, typename L1, typename L2, typename L3, int dim>
void For (BoxND<dim> const& box1, T1 ncomp1, L1&& f1,
          BoxND<dim> const& box2, T2 ncomp2, L2&& f2,
          BoxND<dim> const& box3, T3 ncomp3, L3&& f3) noexcept
{
    ParallelFor<MT>(Gpu::KernelInfo{},
                box1,ncomp1,std::forward<L1>(f1),
                box2,ncomp2,std::forward<L2>(f2),
                box3,ncomp3,std::forward<L3>(f3));
}

template <std::integral T, typename L>
requires (MaybeHostDeviceRunnable<L>::value)
void
HostDeviceParallelFor (Gpu::KernelInfo const& info, T n, L&& f)
{
    if (Gpu::inLaunchRegion()) {
        ParallelFor<AMREX_GPU_MAX_THREADS>(info,n,std::forward<L>(f));
    } else {
        AMREX_PRAGMA_SIMD
        for (T i = 0; i < n; ++i) { f(i); }
    }
}

template <int MT, std::integral T, typename L>
requires (MaybeHostDeviceRunnable<L>::value)
void
HostDeviceParallelFor (Gpu::KernelInfo const& info, T n, L&& f)
{
    if (Gpu::inLaunchRegion()) {
        ParallelFor<MT>(info,n,std::forward<L>(f));
    } else {
        AMREX_PRAGMA_SIMD
        for (T i = 0; i < n; ++i) { f(i); }
    }
}

template <std::integral T, typename L>
requires (MaybeHostDeviceRunnable<L>::value)
void
HostDeviceParallelFor (T n, L&& f) noexcept
{
    HostDeviceParallelFor<AMREX_GPU_MAX_THREADS>(Gpu::KernelInfo{}, n, std::forward<L>(f));
}

template <int MT, std::integral T, typename L>
requires (MaybeHostDeviceRunnable<L>::value)
void
HostDeviceParallelFor (T n, L&& f) noexcept
{
    HostDeviceParallelFor<MT>(Gpu::KernelInfo{}, n, std::forward<L>(f));
}

template <typename L, int dim>
requires (MaybeHostDeviceRunnable<L>::value)
void
HostDeviceParallelFor (Gpu::KernelInfo const& info, BoxND<dim> const& box, L&& f)
{
    if (Gpu::inLaunchRegion()) {
        ParallelFor<AMREX_GPU_MAX_THREADS>(info, box,std::forward<L>(f));
    } else {
        LoopConcurrentOnCpu(box,std::forward<L>(f));
    }
}

template <int MT, typename L, int dim>
requires (MaybeHostDeviceRunnable<L>::value)
void
HostDeviceParallelFor (Gpu::KernelInfo const& info, BoxND<dim> const& box, L&& f)
{
    if (Gpu::inLaunchRegion()) {
        ParallelFor<MT>(info, box,std::forward<L>(f));
    } else {
        LoopConcurrentOnCpu(box,std::forward<L>(f));
    }
}

template <std::integral T, typename L, int dim>
requires (MaybeHostDeviceRunnable<L>::value)
void
HostDeviceParallelFor (Gpu::KernelInfo const& info, BoxND<dim> const& box, T ncomp, L&& f)
{
    if (Gpu::inLaunchRegion()) {
        ParallelFor<AMREX_GPU_MAX_THREADS>(info, box,ncomp,std::forward<L>(f));
    } else {
        LoopConcurrentOnCpu(box,ncomp,std::forward<L>(f));
    }
}

template <int MT, std::integral T, typename L, int dim>
requires (MaybeHostDeviceRunnable<L>::value)
void
HostDeviceParallelFor (Gpu::KernelInfo const& info, BoxND<dim> const& box, T ncomp, L&& f)
{
    if (Gpu::inLaunchRegion()) {
        ParallelFor<MT>(info, box,ncomp,std::forward<L>(f));
    } else {
        LoopConcurrentOnCpu(box,ncomp,std::forward<L>(f));
    }
}

template <typename L1, typename L2, int dim>
requires (MaybeHostDeviceRunnable<L1>::value && MaybeHostDeviceRunnable<L2>::value)
void
HostDeviceParallelFor (Gpu::KernelInfo const& info,
                       BoxND<dim> const& box1, BoxND<dim> const& box2, L1&& f1, L2&& f2)
{
    if (Gpu::inLaunchRegion()) {
        ParallelFor<AMREX_GPU_MAX_THREADS>(info,box1,box2,std::forward<L1>(f1),std::forward<L2>(f2));
    } else {
        LoopConcurrentOnCpu(box1,std::forward<L1>(f1));
        LoopConcurrentOnCpu(box2,std::forward<L2>(f2));
    }
}

template <int MT, typename L1, typename L2, int dim>
requires (MaybeHostDeviceRunnable<L1>::value && MaybeHostDeviceRunnable<L2>::value)
void
HostDeviceParallelFor (Gpu::KernelInfo const& info,
                       BoxND<dim> const& box1, BoxND<dim> const& box2, L1&& f1, L2&& f2)
{
    if (Gpu::inLaunchRegion()) {
        ParallelFor<MT>(info,box1,box2,std::forward<L1>(f1),std::forward<L2>(f2));
    } else {
        LoopConcurrentOnCpu(box1,std::forward<L1>(f1));
        LoopConcurrentOnCpu(box2,std::forward<L2>(f2));
    }
}

template <typename L1, typename L2, typename L3, int dim>
requires (MaybeHostDeviceRunnable<L1>::value && MaybeHostDeviceRunnable<L2>::value &&
          MaybeHostDeviceRunnable<L3>::value)
void
HostDeviceParallelFor (Gpu::KernelInfo const& info,
                       BoxND<dim> const& box1, BoxND<dim> const& box2, BoxND<dim> const& box3,
                       L1&& f1, L2&& f2, L3&& f3)
{
    if (Gpu::inLaunchRegion()) {
        ParallelFor<AMREX_GPU_MAX_THREADS>(info,box1,box2,box3,
                    std::forward<L1>(f1),std::forward<L2>(f2),std::forward<L3>(f3));
    } else {
        LoopConcurrentOnCpu(box1,std::forward<L1>(f1));
        LoopConcurrentOnCpu(box2,std::forward<L2>(f2));
        LoopConcurrentOnCpu(box3,std::forward<L3>(f3));
    }
}

template <int MT, typename L1, typename L2, typename L3, int dim>
requires (MaybeHostDeviceRunnable<L1>::value && MaybeHostDeviceRunnable<L2>::value &&
          MaybeHostDeviceRunnable<L3>::value)
void
HostDeviceParallelFor (Gpu::KernelInfo const& info,
                       BoxND<dim> const& box1, BoxND<dim> const& box2, BoxND<dim> const& box3,
                       L1&& f1, L2&& f2, L3&& f3)
{
    if (Gpu::inLaunchRegion()) {
        ParallelFor<MT>(info,box1,box2,box3,
                    std::forward<L1>(f1),std::forward<L2>(f2),std::forward<L3>(f3));
    } else {
        LoopConcurrentOnCpu(box1,std::forward<L1>(f1));
        LoopConcurrentOnCpu(box2,std::forward<L2>(f2));
        LoopConcurrentOnCpu(box3,std::forward<L3>(f3));
    }
}

template <std::integral T1, std::integral T2, typename L1, typename L2, int dim>
requires (MaybeHostDeviceRunnable<L1>::value && MaybeHostDeviceRunnable<L2>::value)
void
HostDeviceParallelFor (Gpu::KernelInfo const& info,
                       BoxND<dim> const& box1, T1 ncomp1, L1&& f1,
                       BoxND<dim> const& box2, T2 ncomp2, L2&& f2)
{
    if (Gpu::inLaunchRegion()) {
        ParallelFor<AMREX_GPU_MAX_THREADS>(info,box1,ncomp1,std::forward<L1>(f1),box2,ncomp2,std::forward<L2>(f2));
    } else {
        LoopConcurrentOnCpu(box1,ncomp1,std::forward<L1>(f1));
        LoopConcurrentOnCpu(box2,ncomp2,std::forward<L2>(f2));
    }
}

template <int MT, std::integral T1, std::integral T2, typename L1, typename L2, int dim>
requires (MaybeHostDeviceRunnable<L1>::value && MaybeHostDeviceRunnable<L2>::value)
void
HostDeviceParallelFor (Gpu::KernelInfo const& info,
                       BoxND<dim> const& box1, T1 ncomp1, L1&& f1,
                       BoxND<dim> const& box2, T2 ncomp2, L2&& f2)
{
    if (Gpu::inLaunchRegion()) {
        ParallelFor<MT>(info,box1,ncomp1,std::forward<L1>(f1),box2,ncomp2,std::forward<L2>(f2));
    } else {
        LoopConcurrentOnCpu(box1,ncomp1,std::forward<L1>(f1));
        LoopConcurrentOnCpu(box2,ncomp2,std::forward<L2>(f2));
    }
}

template <std::integral T1, std::integral T2, std::integral T3, typename L1, typename L2, typename L3, int dim>
requires (MaybeHostDeviceRunnable<L1>::value && MaybeHostDeviceRunnable<L2>::value &&
          MaybeHostDeviceRunnable<L3>::value)
void
HostDeviceParallelFor (Gpu::KernelInfo const& info,
                       BoxND<dim> const& box1, T1 ncomp1, L1&& f1,
                       BoxND<dim> const& box2, T2 ncomp2, L2&& f2,
                       BoxND<dim> const& box3, T3 ncomp3, L3&& f3)
{
    if (Gpu::inLaunchRegion()) {
        ParallelFor<AMREX_GPU_MAX_THREADS>(info,
                    box1,ncomp1,std::forward<L1>(f1),
                    box2,ncomp2,std::forward<L2>(f2),
                    box3,ncomp3,std::forward<L3>(f3));
    } else {
        LoopConcurrentOnCpu(box1,ncomp1,std::forward<L1>(f1));
        LoopConcurrentOnCpu(box2,ncomp2,std::forward<L2>(f2));
        LoopConcurrentOnCpu(box3,ncomp3,std::forward<L3>(f3));
    }
}

template <int MT, std::integral T1, std::integral T2, std::integral T3, typename L1, typename L2, typename L3, int dim>
requires (MaybeHostDeviceRunnable<L1>::value && MaybeHostDeviceRunnable<L2>::value &&
          MaybeHostDeviceRunnable<L3>::value)
void
HostDeviceParallelFor (Gpu::KernelInfo const& info,
                       BoxND<dim> const& box1, T1 ncomp1, L1&& f1,
                       BoxND<dim> const& box2, T2 ncomp2, L2&& f2,
                       BoxND<dim> const& box3, T3 ncomp3, L3&& f3)
{
    if (Gpu::inLaunchRegion()) {
        ParallelFor<MT>(info,
                    box1,ncomp1,std::forward<L1>(f1),
                    box2,ncomp2,std::forward<L2>(f2),
                    box3,ncomp3,std::forward<L3>(f3));
    } else {
        LoopConcurrentOnCpu(box1,ncomp1,std::forward<L1>(f1));
        LoopConcurrentOnCpu(box2,ncomp2,std::forward<L2>(f2));
        LoopConcurrentOnCpu(box3,ncomp3,std::forward<L3>(f3));
    }
}

template <std::integral T, typename L>
void HostDeviceFor (Gpu::KernelInfo const& info, T n, L&& f) noexcept
{
    HostDeviceParallelFor<AMREX_GPU_MAX_THREADS>(info,n,std::forward<L>(f));
}

template <int MT, std::integral T, typename L>
void HostDeviceFor (Gpu::KernelInfo const& info, T n, L&& f) noexcept
{
    HostDeviceParallelFor<MT>(info,n,std::forward<L>(f));
}

template <typename L, int dim>
void HostDeviceFor (Gpu::KernelInfo const& info, BoxND<dim> const& box, L&& f) noexcept
{
    HostDeviceParallelFor<AMREX_GPU_MAX_THREADS>(info,box,std::forward<L>(f));
}

template <int MT, typename L, int dim>
void HostDeviceFor (Gpu::KernelInfo const& info, BoxND<dim> const& box, L&& f) noexcept
{
    HostDeviceParallelFor<MT>(info,box,std::forward<L>(f));
}

template <std::integral T, typename L, int dim>
void HostDeviceFor (Gpu::KernelInfo const& info, BoxND<dim> const& box, T ncomp, L&& f) noexcept
{
    HostDeviceParallelFor<AMREX_GPU_MAX_THREADS>(info,box,ncomp,std::forward<L>(f));
}

template <int MT, std::integral T, typename L, int dim>
void HostDeviceFor (Gpu::KernelInfo const& info, BoxND<dim> const& box, T ncomp, L&& f) noexcept
{
    HostDeviceParallelFor<MT>(info,box,ncomp,std::forward<L>(f));
}

template <typename L1, typename L2, int dim>
void HostDeviceFor (Gpu::KernelInfo const& info,
                    BoxND<dim> const& box1, BoxND<dim> const& box2, L1&& f1, L2&& f2) noexcept
{
    HostDeviceParallelFor<AMREX_GPU_MAX_THREADS>(info,box1,box2,std::forward<L1>(f1),std::forward<L2>(f2));
}

template <int MT, typename L1, typename L2, int dim>
void HostDeviceFor (Gpu::KernelInfo const& info,
                    BoxND<dim> const& box1, BoxND<dim> const& box2, L1&& f1, L2&& f2) noexcept
{
    HostDeviceParallelFor<MT>(info,box1,box2,std::forward<L1>(f1),std::forward<L2>(f2));
}

template <typename L1, typename L2, typename L3, int dim>
void HostDeviceFor (Gpu::KernelInfo const& info,
                    BoxND<dim> const& box1, BoxND<dim> const& box2, BoxND<dim> const& box3,
                    L1&& f1, L2&& f2, L3&& f3) noexcept
{
    HostDeviceParallelFor<AMREX_GPU_MAX_THREADS>(info, box1,box2,box3,
                          std::forward<L1>(f1),std::forward<L2>(f2),std::forward<L3>(f3));
}

template <int MT, typename L1, typename L2, typename L3, int dim>
void HostDeviceFor (Gpu::KernelInfo const& info,
                    BoxND<dim> const& box1, BoxND<dim> const& box2, BoxND<dim> const& box3,
                    L1&& f1, L2&& f2, L3&& f3) noexcept
{
    HostDeviceParallelFor<MT>(info, box1,box2,box3,
                          std::forward<L1>(f1),std::forward<L2>(f2),std::forward<L3>(f3));
}

template <std::integral T1, std::integral T2, typename L1, typename L2, int dim>
void HostDeviceFor (Gpu::KernelInfo const& info,
                    BoxND<dim> const& box1, T1 ncomp1, L1&& f1,
                    BoxND<dim> const& box2, T2 ncomp2, L2&& f2) noexcept
{
    HostDeviceParallelFor<AMREX_GPU_MAX_THREADS>(info,box1,ncomp1,std::forward<L1>(f1),box2,ncomp2,std::forward<L2>(f2));
}

template <int MT, std::integral T1, std::integral T2, typename L1, typename L2, int dim>
void HostDeviceFor (Gpu::KernelInfo const& info,
                    BoxND<dim> const& box1, T1 ncomp1, L1&& f1,
                    BoxND<dim> const& box2, T2 ncomp2, L2&& f2) noexcept
{
    HostDeviceParallelFor<MT>(info,box1,ncomp1,std::forward<L1>(f1),box2,ncomp2,std::forward<L2>(f2));
}

template <std::integral T1, std::integral T2, std::integral T3, typename L1, typename L2, typename L3, int dim>
void HostDeviceFor (Gpu::KernelInfo const& info,
                    BoxND<dim> const& box1, T1 ncomp1, L1&& f1,
                    BoxND<dim> const& box2, T2 ncomp2, L2&& f2,
                    BoxND<dim> const& box3, T3 ncomp3, L3&& f3) noexcept
{
    HostDeviceParallelFor<AMREX_GPU_MAX_THREADS>(info,
                          box1,ncomp1,std::forward<L1>(f1),
                          box2,ncomp2,std::forward<L2>(f2),
                          box3,ncomp3,std::forward<L3>(f3));
}

template <int MT, std::integral T1, std::integral T2, std::integral T3, typename L1, typename L2, typename L3, int dim>
void HostDeviceFor (Gpu::KernelInfo const& info,
                    BoxND<dim> const& box1, T1 ncomp1, L1&& f1,
                    BoxND<dim> const& box2, T2 ncomp2, L2&& f2,
                    BoxND<dim> const& box3, T3 ncomp3, L3&& f3) noexcept
{
    HostDeviceParallelFor<MT>(info,
                          box1,ncomp1,std::forward<L1>(f1),
                          box2,ncomp2,std::forward<L2>(f2),
                          box3,ncomp3,std::forward<L3>(f3));
}

template <std::integral T, typename L>
void HostDeviceParallelFor (T n, L&& f) noexcept
{
    HostDeviceParallelFor<AMREX_GPU_MAX_THREADS>(Gpu::KernelInfo{},n,std::forward<L>(f));
}

template <int MT, std::integral T, typename L>
void HostDeviceParallelFor (T n, L&& f) noexcept
{
    HostDeviceParallelFor<MT>(Gpu::KernelInfo{},n,std::forward<L>(f));
}

template <typename L, int dim>
void HostDeviceParallelFor (BoxND<dim> const& box, L&& f) noexcept
{
    HostDeviceParallelFor<AMREX_GPU_MAX_THREADS>(Gpu::KernelInfo{},box,std::forward<L>(f));
}

template <int MT, typename L, int dim>
void HostDeviceParallelFor (BoxND<dim> const& box, L&& f) noexcept
{
    HostDeviceParallelFor<MT>(Gpu::KernelInfo{},box,std::forward<L>(f));
}

template <std::integral T, typename L, int dim>
void HostDeviceParallelFor (BoxND<dim> const& box, T ncomp, L&& f) noexcept
{
    HostDeviceParallelFor<AMREX_GPU_MAX_THREADS>(Gpu::KernelInfo{},box,ncomp,std::forward<L>(f));
}

template <int MT, std::integral T, typename L, int dim>
void HostDeviceParallelFor (BoxND<dim> const& box, T ncomp, L&& f) noexcept
{
    HostDeviceParallelFor<MT>(Gpu::KernelInfo{},box,ncomp,std::forward<L>(f));
}

template <typename L1, typename L2, int dim>
void HostDeviceParallelFor (BoxND<dim> const& box1, BoxND<dim> const& box2, L1&& f1, L2&& f2) noexcept
{
    HostDeviceParallelFor<AMREX_GPU_MAX_THREADS>(Gpu::KernelInfo{},box1,box2,std::forward<L1>(f1),std::forward<L2>(f2));
}

template <int MT, typename L1, typename L2, int dim>
void HostDeviceParallelFor (BoxND<dim> const& box1, BoxND<dim> const& box2, L1&& f1, L2&& f2) noexcept
{
    HostDeviceParallelFor<MT>(Gpu::KernelInfo{},box1,box2,std::forward<L1>(f1),std::forward<L2>(f2));
}

template <typename L1, typename L2, typename L3, int dim>
void HostDeviceParallelFor (BoxND<dim> const& box1, BoxND<dim> const& box2, BoxND<dim> const& box3,
                            L1&& f1, L2&& f2, L3&& f3) noexcept
{
    HostDeviceParallelFor<AMREX_GPU_MAX_THREADS>(Gpu::KernelInfo{}, box1,box2,box3,
                          std::forward<L1>(f1),std::forward<L2>(f2),std::forward<L3>(f3));
}

template <int MT, typename L1, typename L2, typename L3, int dim>
void HostDeviceParallelFor (BoxND<dim> const& box1, BoxND<dim> const& box2, BoxND<dim> const& box3,
                            L1&& f1, L2&& f2, L3&& f3) noexcept
{
    HostDeviceParallelFor<MT>(Gpu::KernelInfo{}, box1,box2,box3,
                          std::forward<L1>(f1),std::forward<L2>(f2),std::forward<L3>(f3));
}

template <std::integral T1, std::integral T2, typename L1, typename L2, int dim>
void HostDeviceParallelFor (BoxND<dim> const& box1, T1 ncomp1, L1&& f1,
                            BoxND<dim> const& box2, T2 ncomp2, L2&& f2) noexcept
{
    HostDeviceParallelFor<AMREX_GPU_MAX_THREADS>(Gpu::KernelInfo{},box1,ncomp1,std::forward<L1>(f1),box2,ncomp2,std::forward<L2>(f2));
}

template <int MT, std::integral T1, std::integral T2, typename L1, typename L2, int dim>
void HostDeviceParallelFor (BoxND<dim> const& box1, T1 ncomp1, L1&& f1,
                            BoxND<dim> const& box2, T2 ncomp2, L2&& f2) noexcept
{
    HostDeviceParallelFor<MT>(Gpu::KernelInfo{},box1,ncomp1,std::forward<L1>(f1),box2,ncomp2,std::forward<L2>(f2));
}

template <std::integral T1, std::integral T2, std::integral T3, typename L1, typename L2, typename L3, int dim>
void HostDeviceParallelFor (BoxND<dim> const& box1, T1 ncomp1, L1&& f1,
                            BoxND<dim> const& box2, T2 ncomp2, L2&& f2,
                            BoxND<dim> const& box3, T3 ncomp3, L3&& f3) noexcept
{
    HostDeviceParallelFor<AMREX_GPU_MAX_THREADS>(Gpu::KernelInfo{},
                          box1,ncomp1,std::forward<L1>(f1),
                          box2,ncomp2,std::forward<L2>(f2),
                          box3,ncomp3,std::forward<L3>(f3));
}

template <int MT, std::integral T1, std::integral T2, std::integral T3, typename L1, typename L2, typename L3, int dim>
void HostDeviceParallelFor (BoxND<dim> const& box1, T1 ncomp1, L1&& f1,
                            BoxND<dim> const& box2, T2 ncomp2, L2&& f2,
                            BoxND<dim> const& box3, T3 ncomp3, L3&& f3) noexcept
{
    HostDeviceParallelFor<MT>(Gpu::KernelInfo{},
                          box1,ncomp1,std::forward<L1>(f1),
                          box2,ncomp2,std::forward<L2>(f2),
                          box3,ncomp3,std::forward<L3>(f3));
}

}

#endif
