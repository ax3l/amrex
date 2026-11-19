#ifndef AMREX_GPU_LAUNCH_FUNCTS_KOKKOS_H_
#define AMREX_GPU_LAUNCH_FUNCTS_KOKKOS_H_
#include <AMReX_Config.H>

/*
 * Element-wise kernel launches (ParallelFor, ParallelForRNG, launch) for
 * GPU builds, implemented with Kokkos.
 *
 * Kernels are Kokkos::TeamPolicy launches with team size MT (the AMReX
 * "block size") on the Kokkos execution space instance of the current AMReX
 * GPU stream.  Each team member handles one element, as AMReX's native
 * kernels did, so the 1D index decomposition of Boxes (BoxIndexer) and the
 * per-thread random number states keep their meaning.
 *
 * A TeamPolicy (rather than a RangePolicy) is used for all of them, because
 * whether a device lambda takes a Gpu::Handler is only known in device code
 * (nvcc cannot introspect __device__ lambdas on the host), and Handler-based
 * block reductions need the team.  On CUDA, TeamPolicy and RangePolicy
 * kernels with the same work decomposition perform the same.
 */

#include <AMReX_Kokkos.H>

namespace amrex {

/// \cond DOXYGEN_IGNORE
namespace detail {

    using KokkosTeamMember = Kokkos::TeamPolicy<Gpu::KokkosExecSpace>::member_type;

    template <int MT>
    using KokkosTeamPolicy = Kokkos::TeamPolicy<Gpu::KokkosExecSpace, Kokkos::LaunchBounds<MT,1>>;

    template <int MT>
    KokkosTeamPolicy<MT> kokkos_team_policy (int nblocks)
    {
        return KokkosTeamPolicy<MT>(Gpu::kokkosExecSpace(), nblocks, MT);
    }

    // global thread index of a team member
    template <int MT>
    AMREX_GPU_DEVICE AMREX_FORCE_INLINE
    std::uint64_t kokkos_global_tid (KokkosTeamMember const& tm) noexcept
    {
        return std::uint64_t(MT)*std::uint64_t(tm.league_rank()) + std::uint64_t(tm.team_rank());
    }

    template <int MT, typename T, typename F>
    struct KokkosLaunchN
    {
        F f;
        T start_idx;
        T nleft;
        AMREX_GPU_DEVICE void operator() (KokkosTeamMember const& tm) const noexcept {
            auto tid = kokkos_global_tid<MT>(tm);
            if (tid < std::uint64_t(nleft)) {
                f(T(tid)+start_idx);
            }
        }
    };

    template <int MT, int dim, typename F>
    struct KokkosLaunchBox
    {
        F f;
        BoxIndexerND<dim> indexer;
        IndexTypeND<dim> type;
        typename BoxIndexerND<dim>::UInt start_idx;
        AMREX_GPU_DEVICE void operator() (KokkosTeamMember const& tm) const noexcept {
            using UInt = typename BoxIndexerND<dim>::UInt;
            auto icell = UInt(kokkos_global_tid<MT>(tm)) + start_idx;
            if (icell < indexer.numPts()) {
                auto iv = indexer.intVect(icell);
                f(BoxND<dim>(iv,iv,type));
            }
        }
    };

    template <int MT, typename T, typename F>
    struct KokkosParForN
    {
        F f;
        T start_idx;
        T nleft;
        AMREX_GPU_DEVICE void operator() (KokkosTeamMember const& tm) const noexcept {
            auto tid = kokkos_global_tid<MT>(tm);
            if (tid < std::uint64_t(nleft)) {
                call_f_scalar_handler(f, T(tid)+start_idx,
                    Gpu::Handler(int(amrex::min(std::uint64_t(nleft)-tid+std::uint64_t(tm.team_rank()),
                                                std::uint64_t(MT))), &tm));
            }
        }
    };

    template <int MT, int dim, typename F>
    struct KokkosParForBox
    {
        F f;
        BoxIndexerND<dim> indexer;
        typename BoxIndexerND<dim>::UInt start_idx;
        AMREX_GPU_DEVICE void operator() (KokkosTeamMember const& tm) const noexcept {
            using UInt = typename BoxIndexerND<dim>::UInt;
            auto icell = UInt(kokkos_global_tid<MT>(tm)) + start_idx;
            if (icell < indexer.numPts()) {
                auto iv = indexer.intVect(icell);
                call_f_intvect_handler(f, iv,
                    Gpu::Handler(int(amrex::min(indexer.numPts()-icell+UInt(tm.team_rank()), UInt(MT))), &tm));
            }
        }
    };

    template <int MT, int dim, typename T, typename F>
    struct KokkosParForBoxN
    {
        F f;
        BoxIndexerND<dim> indexer;
        T ncomp;
        typename BoxIndexerND<dim>::UInt start_idx;
        AMREX_GPU_DEVICE void operator() (KokkosTeamMember const& tm) const noexcept {
            using UInt = typename BoxIndexerND<dim>::UInt;
            auto icell = UInt(kokkos_global_tid<MT>(tm)) + start_idx;
            if (icell < indexer.numPts()) {
                auto iv = indexer.intVect(icell);
                call_f_intvect_ncomp_handler(f, iv, ncomp,
                    Gpu::Handler(int(amrex::min(indexer.numPts()-icell+UInt(tm.team_rank()), UInt(MT))), &tm));
            }
        }
    };

    // Grid-stride loop over [0,n) with one RandomEngine per global thread
    template <int MT, typename T, typename F>
    struct KokkosParForRNGN
    {
        F f;
        T n;
        randState_t* rand_state;
        AMREX_GPU_DEVICE void operator() (KokkosTeamMember const& tm) const noexcept {
            auto const tid = kokkos_global_tid<MT>(tm);
            RandomEngine engine{&(rand_state[tid])};
            auto const stride = std::uint64_t(MT)*std::uint64_t(tm.league_size());
            for (std::uint64_t i = tid; i < std::uint64_t(n); i += stride) {
                f(T(i),engine);
            }
        }
    };

    template <int MT, int dim, typename F>
    struct KokkosParForRNGBox
    {
        F f;
        BoxIndexerND<dim> indexer;
        randState_t* rand_state;
        AMREX_GPU_DEVICE void operator() (KokkosTeamMember const& tm) const noexcept {
            using UInt = typename BoxIndexerND<dim>::UInt;
            auto const tid = kokkos_global_tid<MT>(tm);
            RandomEngine engine{&(rand_state[tid])};
            auto const stride = UInt(MT)*UInt(tm.league_size());
            for (UInt icell = UInt(tid); icell < indexer.numPts(); icell += stride) {
                auto iv = indexer.intVect(icell);
                call_f_intvect_engine(f, iv, engine);
            }
        }
    };

    template <int MT, int dim, typename T, typename F>
    struct KokkosParForRNGBoxN
    {
        F f;
        BoxIndexerND<dim> indexer;
        T ncomp;
        randState_t* rand_state;
        AMREX_GPU_DEVICE void operator() (KokkosTeamMember const& tm) const noexcept {
            using UInt = typename BoxIndexerND<dim>::UInt;
            auto const tid = kokkos_global_tid<MT>(tm);
            RandomEngine engine{&(rand_state[tid])};
            auto const stride = UInt(MT)*UInt(tm.league_size());
            for (UInt icell = UInt(tid); icell < indexer.numPts(); icell += stride) {
                auto iv = indexer.intVect(icell);
                call_f_intvect_ncomp_engine(f, iv, ncomp, engine);
            }
        }
    };

    // Fused launch over several boxes.  NC is void for f(i,j,k) and an
    // integer type for f(i,j,k,n).
    template <int MT, int dim, typename F1, typename F2, typename F3, typename NC>
    struct KokkosParForFused
    {
        F1 f1;
        F2 f2;
        F3 f3;
        BoxIndexerND<dim> indexer1;
        BoxIndexerND<dim> indexer2;
        BoxIndexerND<dim> indexer3;
        std::conditional_t<std::is_void_v<NC>, int, NC> ncomp1;
        std::conditional_t<std::is_void_v<NC>, int, NC> ncomp2;
        std::conditional_t<std::is_void_v<NC>, int, NC> ncomp3;
        int nboxes;

        template <typename F, typename N>
        AMREX_GPU_DEVICE AMREX_FORCE_INLINE
        static void call (F const& f, BoxIndexerND<dim> const& indexer,
                          typename BoxIndexerND<dim>::UInt icell, [[maybe_unused]] N ncomp) noexcept
        {
            if (icell < indexer.numPts()) {
                auto iv = indexer.intVect(icell);
                if constexpr (std::is_void_v<NC>) {
                    call_f_intvect(f, iv);
                } else {
                    call_f_intvect_ncomp(f, iv, ncomp);
                }
            }
        }

        AMREX_GPU_DEVICE void operator() (KokkosTeamMember const& tm) const noexcept {
            using UInt = typename BoxIndexerND<dim>::UInt;
            auto const ncells = amrex::max(indexer1.numPts(), indexer2.numPts(),
                                           (nboxes == 3) ? indexer3.numPts() : UInt(0));
            auto const stride = UInt(MT)*UInt(tm.league_size());
            for (UInt icell = UInt(kokkos_global_tid<MT>(tm)); icell < ncells; icell += stride) {
                call(f1, indexer1, icell, ncomp1);
                call(f2, indexer2, icell, ncomp2);
                if constexpr (!std::is_same_v<F3,int>) {
                    call(f3, indexer3, icell, ncomp3);
                }
            }
        }
    };

    template <int MT, typename Functor>
    void kokkos_launch (int nblocks, Functor const& functor)
    {
        Kokkos::parallel_for(kokkos_team_policy<MT>(nblocks), functor);
    }

    template <typename F>
    struct KokkosTeamFunctor
    {
        F f;
        AMREX_GPU_DEVICE void operator() (KokkosTeamMember const& tm) const noexcept { f(tm); }
    };
}
/// \endcond

namespace Gpu {

/**
 * \brief Launch a kernel of `nblocks` Kokkos teams (blocks) of MT threads.
 *
 * `f(tm)` is called by every thread with its Gpu::KokkosTeamMember `tm`:
 * tm.league_rank() is the block index and tm.team_rank() the thread index
 * in the block.  Consecutive team ranks are consecutive threads of a warp.
 * `shared_mem_bytes` bytes of team scratch memory (GPU shared memory) are
 * available through tm.team_shmem().get_shmem(shared_mem_bytes).  The
 * kernel runs on the current AMReX GPU stream.
 */
template <int MT, typename F>
void team_launch (int nblocks, std::size_t shared_mem_bytes, F const& f)
{
    if (nblocks <= 0) { return; }
    auto policy = amrex::detail::kokkos_team_policy<MT>(nblocks);
    if (shared_mem_bytes > 0) {
        policy.set_scratch_size(0, Kokkos::PerTeam(shared_mem_bytes));
    }
    Kokkos::parallel_for(policy, amrex::detail::KokkosTeamFunctor<F>{f});
    AMREX_GPU_ERROR_CHECK();
}

//! Gpu::team_launch without team scratch memory
template <int MT, typename F>
void team_launch (int nblocks, F const& f)
{
    team_launch<MT>(nblocks, 0, f);
}

/**
 * \brief Launch a kernel of `nblocks` blocks of MT threads that take a Gpu::Handler.
 *
 * `f(handler)` is called by every thread.  The handler provides the block
 * and thread indices (blockIdx(), threadIdx(), ...), `shared_mem_bytes`
 * bytes of shared memory (sharedMemory()) and a block barrier
 * (syncThreads()), like the Gpu::Handler of amrex::launch in SYCL builds.
 */
template <int MT, typename F>
void handler_launch (int nblocks, std::size_t shared_mem_bytes, F const& f)
{
    team_launch<MT>(nblocks, shared_mem_bytes,
    [=] AMREX_GPU_DEVICE (KokkosTeamMember const& tm) noexcept
    {
        void* local = (shared_mem_bytes > 0) ? tm.team_scratch(0).get_shmem(shared_mem_bytes) : nullptr;
        f(Handler(-1, &tm, local));
    });
}

}

template<int MT, std::integral T, typename L>
void launch (T const& n, L const& f)
{
    static_assert(sizeof(T) >= 2);
    if (amrex::isEmpty(n)) { return; }
    const auto& nec = Gpu::makeNExecutionConfigs<MT>(n);
    for (auto const& ec : nec) {
        const T start_idx = T(ec.start_idx);
        detail::kokkos_launch<MT>(ec.nblocks, detail::KokkosLaunchN<MT,T,L>{f, start_idx, T(n-start_idx)});
    }
    AMREX_GPU_ERROR_CHECK();
}

template<int MT, int dim, typename L>
void launch (BoxND<dim> const& box, L const& f)
{
    if (box.isEmpty()) { return; }
    const auto& nec = Gpu::makeNExecutionConfigs<MT>(box);
    const BoxIndexerND<dim> indexer(box);
    using UInt = typename BoxIndexerND<dim>::UInt;
    for (auto const& ec : nec) {
        detail::kokkos_launch<MT>(ec.nblocks, detail::KokkosLaunchBox<MT,dim,L>
                                  {f, indexer, box.ixType(), UInt(ec.start_idx)});
    }
    AMREX_GPU_ERROR_CHECK();
}

/**
 * \ingroup amrex_execution
 * \brief Performance-portable kernel launch function
 */
template <int MT, std::integral T, typename L>
requires (MaybeDeviceRunnable<L>::value)
void
ParallelFor (Gpu::KernelInfo const&, T n, L const& f)
{
    static_assert(sizeof(T) >= 2);
    if (amrex::isEmpty(n)) { return; }
    const auto& nec = Gpu::makeNExecutionConfigs<MT>(n);
    for (auto const& ec : nec) {
        const T start_idx = T(ec.start_idx);
        detail::kokkos_launch<MT>(ec.nblocks, detail::KokkosParForN<MT,T,L>{f, start_idx, T(n-start_idx)});
    }
    AMREX_GPU_ERROR_CHECK();
}

/**
 * \ingroup amrex_execution
 * \brief Performance-portable kernel launch function
 */
template <int MT, typename L, int dim>
requires (MaybeDeviceRunnable<L>::value)
void
ParallelFor (Gpu::KernelInfo const&, BoxND<dim> const& box, L const& f)
{
    if (amrex::isEmpty(box)) { return; }
    const BoxIndexerND<dim> indexer(box);
    using UInt = typename BoxIndexerND<dim>::UInt;
    const auto& nec = Gpu::makeNExecutionConfigs<MT>(box);
    for (auto const& ec : nec) {
        detail::kokkos_launch<MT>(ec.nblocks, detail::KokkosParForBox<MT,dim,L>{f, indexer, UInt(ec.start_idx)});
    }
    AMREX_GPU_ERROR_CHECK();
}

/**
 * \ingroup amrex_execution
 * \brief Performance-portable kernel launch function
 */
template <int MT, std::integral T, typename L, int dim>
requires (MaybeDeviceRunnable<L>::value)
void
ParallelFor (Gpu::KernelInfo const&, BoxND<dim> const& box, T ncomp, L const& f)
{
    if (amrex::isEmpty(box)) { return; }
    const BoxIndexerND<dim> indexer(box);
    using UInt = typename BoxIndexerND<dim>::UInt;
    const auto& nec = Gpu::makeNExecutionConfigs<MT>(box);
    for (auto const& ec : nec) {
        detail::kokkos_launch<MT>(ec.nblocks, detail::KokkosParForBoxN<MT,dim,T,L>{f, indexer, ncomp, UInt(ec.start_idx)});
    }
    AMREX_GPU_ERROR_CHECK();
}

/**
 * \ingroup amrex_execution
 * \brief Performance-portable kernel launch function with random number
 * generation support.
 */
template <std::integral T, typename L>
requires (MaybeDeviceRunnable<L>::value)
void
ParallelForRNG (T n, L const& f)
{
    if (amrex::isEmpty(n)) { return; }
    randState_t* rand_state = getRandState();
    const auto ec = Gpu::ExecutionConfig(n);
    int nblocks = int(amrex::min(ec.numBlocks.x, Gpu::Device::maxBlocksPerLaunch()));
    detail::kokkos_launch<AMREX_GPU_MAX_THREADS>(nblocks,
        detail::KokkosParForRNGN<AMREX_GPU_MAX_THREADS,T,L>{f, n, rand_state});
    Gpu::streamSynchronize(); // To avoid multiple streams using RNG
    AMREX_GPU_ERROR_CHECK();
}

/**
 * \ingroup amrex_execution
 * \brief Performance-portable kernel launch function with random number
 * generation support.
 */
template <typename L, int dim>
requires (MaybeDeviceRunnable<L>::value)
void
ParallelForRNG (BoxND<dim> const& box, L const& f)
{
    if (amrex::isEmpty(box)) { return; }
    randState_t* rand_state = getRandState();
    const BoxIndexerND<dim> indexer(box);
    const auto ec = Gpu::ExecutionConfig(box.numPts());
    int nblocks = int(amrex::min(ec.numBlocks.x, Gpu::Device::maxBlocksPerLaunch()));
    detail::kokkos_launch<AMREX_GPU_MAX_THREADS>(nblocks,
        detail::KokkosParForRNGBox<AMREX_GPU_MAX_THREADS,dim,L>{f, indexer, rand_state});
    Gpu::streamSynchronize(); // To avoid multiple streams using RNG
    AMREX_GPU_ERROR_CHECK();
}

/**
 * \ingroup amrex_execution
 * \brief Performance-portable kernel launch function with random number
 * generation support.
 */
template <std::integral T, typename L, int dim>
requires (MaybeDeviceRunnable<L>::value)
void
ParallelForRNG (BoxND<dim> const& box, T ncomp, L const& f)
{
    if (amrex::isEmpty(box)) { return; }
    randState_t* rand_state = getRandState();
    const BoxIndexerND<dim> indexer(box);
    const auto ec = Gpu::ExecutionConfig(box.numPts());
    int nblocks = int(amrex::min(ec.numBlocks.x, Gpu::Device::maxBlocksPerLaunch()));
    detail::kokkos_launch<AMREX_GPU_MAX_THREADS>(nblocks,
        detail::KokkosParForRNGBoxN<AMREX_GPU_MAX_THREADS,dim,T,L>{f, indexer, ncomp, rand_state});
    Gpu::streamSynchronize(); // To avoid multiple streams using RNG
    AMREX_GPU_ERROR_CHECK();
}

/**
 * \ingroup amrex_execution
 * \brief Performance-portable kernel launch function
 */
template <int MT, typename L1, typename L2, int dim>
requires (MaybeDeviceRunnable<L1>::value && MaybeDeviceRunnable<L2>::value)
void
ParallelFor (Gpu::KernelInfo const&,
             BoxND<dim> const& box1, BoxND<dim> const& box2, L1&& f1, L2&& f2)
{
    if (amrex::isEmpty(box1) && amrex::isEmpty(box2)) { return; }
    const auto ec = Gpu::makeExecutionConfig<MT>(std::max(box1.numPts(),box2.numPts()));
    detail::kokkos_launch<MT>(int(ec.numBlocks.x),
        detail::KokkosParForFused<MT,dim,std::decay_t<L1>,std::decay_t<L2>,int,void>
        {f1, f2, 0, BoxIndexerND<dim>(box1), BoxIndexerND<dim>(box2), BoxIndexerND<dim>(box1),
         0, 0, 0, 2});
    AMREX_GPU_ERROR_CHECK();
}

/**
 * \ingroup amrex_execution
 * \brief Performance-portable kernel launch function
 */
template <int MT, typename L1, typename L2, typename L3, int dim>
requires (MaybeDeviceRunnable<L1>::value && MaybeDeviceRunnable<L2>::value &&
          MaybeDeviceRunnable<L3>::value)
void
ParallelFor (Gpu::KernelInfo const&,
             BoxND<dim> const& box1, BoxND<dim> const& box2, BoxND<dim> const& box3,
             L1&& f1, L2&& f2, L3&& f3)
{
    if (amrex::isEmpty(box1) && amrex::isEmpty(box2) && amrex::isEmpty(box3)) { return; }
    const auto ec = Gpu::makeExecutionConfig<MT>(std::max({box1.numPts(),box2.numPts(),box3.numPts()}));
    detail::kokkos_launch<MT>(int(ec.numBlocks.x),
        detail::KokkosParForFused<MT,dim,std::decay_t<L1>,std::decay_t<L2>,std::decay_t<L3>,void>
        {f1, f2, f3, BoxIndexerND<dim>(box1), BoxIndexerND<dim>(box2), BoxIndexerND<dim>(box3),
         0, 0, 0, 3});
    AMREX_GPU_ERROR_CHECK();
}

/**
 * \ingroup amrex_execution
 * \brief Performance-portable kernel launch function
 */
template <int MT, std::integral T1, std::integral T2, typename L1, typename L2, int dim>
requires (MaybeDeviceRunnable<L1>::value && MaybeDeviceRunnable<L2>::value)
void
ParallelFor (Gpu::KernelInfo const&,
             BoxND<dim> const& box1, T1 ncomp1, L1&& f1,
             BoxND<dim> const& box2, T2 ncomp2, L2&& f2)
{
    if (amrex::isEmpty(box1) && amrex::isEmpty(box2)) { return; }
    using NC = std::common_type_t<T1,T2>;
    const auto ec = Gpu::makeExecutionConfig<MT>(std::max(box1.numPts(),box2.numPts()));
    detail::kokkos_launch<MT>(int(ec.numBlocks.x),
        detail::KokkosParForFused<MT,dim,std::decay_t<L1>,std::decay_t<L2>,int,NC>
        {f1, f2, 0, BoxIndexerND<dim>(box1), BoxIndexerND<dim>(box2), BoxIndexerND<dim>(box1),
         NC(ncomp1), NC(ncomp2), NC(0), 2});
    AMREX_GPU_ERROR_CHECK();
}

/**
 * \ingroup amrex_execution
 * \brief Performance-portable kernel launch function
 */
template <int MT, std::integral T1, std::integral T2, std::integral T3, typename L1, typename L2, typename L3, int dim>
requires (MaybeDeviceRunnable<L1>::value && MaybeDeviceRunnable<L2>::value &&
          MaybeDeviceRunnable<L3>::value)
void
ParallelFor (Gpu::KernelInfo const&,
             BoxND<dim> const& box1, T1 ncomp1, L1&& f1,
             BoxND<dim> const& box2, T2 ncomp2, L2&& f2,
             BoxND<dim> const& box3, T3 ncomp3, L3&& f3)
{
    if (amrex::isEmpty(box1) && amrex::isEmpty(box2) && amrex::isEmpty(box3)) { return; }
    using NC = std::common_type_t<T1,T2,T3>;
    const auto ec = Gpu::makeExecutionConfig<MT>(std::max({box1.numPts(),box2.numPts(),box3.numPts()}));
    detail::kokkos_launch<MT>(int(ec.numBlocks.x),
        detail::KokkosParForFused<MT,dim,std::decay_t<L1>,std::decay_t<L2>,std::decay_t<L3>,NC>
        {f1, f2, f3, BoxIndexerND<dim>(box1), BoxIndexerND<dim>(box2), BoxIndexerND<dim>(box3),
         NC(ncomp1), NC(ncomp2), NC(ncomp3), 3});
    AMREX_GPU_ERROR_CHECK();
}

}

#endif
