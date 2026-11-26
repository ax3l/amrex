#ifndef AMREX_FB_NVSHMEM_H_
#define AMREX_FB_NVSHMEM_H_

/**
 * \file AMReX_FBNvshmem.H
 * \brief Experimental GPU FillBoundary and SumBoundary paths over NVIDIA
 *        NVSHMEM instead of MPI, selected with `amrex.nvshmem.fb_mode` (see
 *        AMReX_NVSHMEM.H).
 *
 * All work is queued on the current GPU stream, which is synchronized once:
 *  1. a short kernel tells the senders that this rank's receive window is
 *     free and waits until the receivers' windows are;
 *  2. the pack kernel writes each message either straight into the receiver's
 *     window (mode 2, where NVSHMEM maps the receiving GPU) or into the local
 *     send window;
 *  3. messages in the send window are put to the receivers from a kernel
 *     (one thread block per message), each with a signal; messages written
 *     directly are followed by a done flag;
 *  4. a short kernel waits for the senders' signals, then the unpack kernel
 *     runs.
 *
 * One such FillBoundary can be in flight at a time.
 */

#if defined(AMREX_USE_NVSHMEM) && defined(AMREX_FB_UNPACK_ON_ARRIVAL)

#include <AMReX_NVSHMEM.H>

#include <nvshmem.h>
#include <nvshmemx.h>

#include <cstdint>

namespace amrex {

/// \cond DOXYGEN_IGNORE
namespace detail {

constexpr int fb_shm_max_peers = 32;

// Ranks with the message number of this exchange.
struct FBShmPeers {
    int pe[fb_shm_max_peers];
    std::uint64_t v[fb_shm_max_peers];
    int n = 0;

    void add (int a_pe, std::uint64_t a_v) {
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(n < fb_shm_max_peers, "NVSHMEM FillBoundary: too many neighbor ranks");
        pe[n] = a_pe; v[n] = a_v; ++n;
    }
};

// Messages to put from the send window into the receivers' windows.
struct FBShmPuts {
    int pe[fb_shm_max_peers];
    std::uint64_t v[fb_shm_max_peers];
    std::size_t loff[fb_shm_max_peers];
    std::size_t roff[fb_shm_max_peers];
    std::size_t nbytes[fb_shm_max_peers];
    int n = 0;

    void add (int a_pe, std::uint64_t a_v, std::size_t a_loff, std::size_t a_roff, std::size_t a_nbytes) {
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(n < fb_shm_max_peers, "NVSHMEM FillBoundary: too many neighbor ranks");
        pe[n] = a_pe; v[n] = a_v; loff[n] = a_loff; roff[n] = a_roff; nbytes[n] = a_nbytes; ++n;
    }
};

// Tell the senders that this rank's window is free for their next message,
// then wait until the receivers' windows are free.
inline void fb_shm_ready (FBShmPeers const& rcv, FBShmPeers const& snd, std::uint64_t* ready, int me)
{
    if (rcv.n == 0 && snd.n == 0) { return; }
    amrex::ParallelFor(1, [=] AMREX_GPU_DEVICE (int) noexcept
    {
        for (int m = 0; m < rcv.n; ++m) { nvshmem_uint64_p(ready + me, rcv.v[m], rcv.pe[m]); }
        for (int m = 0; m < snd.n; ++m) {
            nvshmem_uint64_wait_until(ready + snd.pe[m], NVSHMEM_CMP_GE, snd.v[m]);
        }
    });
}

// Done flags for the messages written directly into the receivers.
inline void fb_shm_done (FBShmPeers const& snd, std::uint64_t* done, int me)
{
    if (snd.n == 0) { return; }
    amrex::ParallelFor(1, [=] AMREX_GPU_DEVICE (int) noexcept
    {
        for (int m = 0; m < snd.n; ++m) { nvshmem_uint64_p(done + me, snd.v[m], snd.pe[m]); }
    });
}

// One thread block per message: put it and set this rank's done word there.
inline void fb_shm_put (FBShmPuts const& p, char* rwin, char const* swin, std::uint64_t* done, int me)
{
    if (p.n == 0) { return; }
    constexpr int nthreads = 512;
    amrex::launch<nthreads>(p.n, Gpu::gpuStream(), [=] AMREX_GPU_DEVICE () noexcept
    {
        const int b = blockIdx.x;
        nvshmemx_putmem_signal_block(rwin + p.roff[b], swin + p.loff[b], p.nbytes[b],
                                     done + me, p.v[b], NVSHMEM_SIGNAL_SET, p.pe[b]);
    });
}

// Wait until every sender has completed its message of this exchange.
inline void fb_shm_wait (FBShmPeers const& rcv, std::uint64_t* done)
{
    if (rcv.n == 0) { return; }
    amrex::ParallelFor(1, [=] AMREX_GPU_DEVICE (int) noexcept
    {
        // not nvshmem_signal_wait_until: in NVSHMEM 3.7 it can return before the
        // data of a network put is visible to this kernel
        for (int m = 0; m < rcv.n; ++m) {
            nvshmem_uint64_wait_until(done + rcv.pe[m], NVSHMEM_CMP_GE, rcv.v[m]);
        }
    });
}

}
/// \endcond

// Learn where this rank's messages go in the receivers' windows and build the
// pack tags: into the receiver where it is mapped, into the send window otherwise.
template <class FAB>
template <typename BUF>
void
FabArray<FAB>::FB_shm_meta (const FB& TheFB, int ncomp, FBOnArrivalMeta<FAB>& e)
{
    BL_PROFILE("FB::shm_meta");

    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(e.rcv.nbytes <= NVSHMEM::windowSize() &&
                                     e.snd.nbytes <= NVSHMEM::windowSize(),
        "NVSHMEM FillBoundary: window too small, raise amrex.nvshmem.window_mb");

    const auto N_rcvs = static_cast<int>(e.rcv.rank.size());
    const auto N_snds = static_cast<int>(e.snd.rank.size());
    MPI_Comm comm = ParallelDescriptor::Communicator();
    const int tag = 27183; // one such message per rank pair and layout, in the same order on both sides

    Vector<unsigned long long> my_off(N_rcvs, 0), peer_off(N_snds, 0);
    Vector<MPI_Request> reqs(N_rcvs, MPI_REQUEST_NULL);
    for (int i = 0; i < N_rcvs; ++i) {
        if (e.rcv.size[i] > 0) {
            my_off[i] = e.rcv.offset[i];
            MPI_Isend(&my_off[i], 1, MPI_UNSIGNED_LONG_LONG, e.rcv.rank[i], tag, comm, &reqs[i]);
        }
    }
    for (int j = 0; j < N_snds; ++j) {
        if (e.snd.size[j] > 0) {
            MPI_Recv(&peer_off[j], 1, MPI_UNSIGNED_LONG_LONG, e.snd.rank[j], tag, comm, MPI_STATUS_IGNORE);
        }
    }
    MPI_Waitall(N_rcvs, reqs.data(), MPI_STATUSES_IGNORE);

    // Same tag order as send_tv, but poff holds the destination address.
    Vector<CommSendBufTag<value_type>> tags;
    e.shm_roff.assign(N_snds, 0);
    e.shm_is_put.assign(N_snds, 1);
    int m = 0;
    for (auto const& kv : *TheFB.m_SndTags) {
        e.shm_roff[m] = peer_off[m];
        char* base = NVSHMEM::peerRecvWindow(kv.first);
        if (base) {
            base += peer_off[m];
            e.shm_is_put[m] = 0;
        } else {
            base = NVSHMEM::sendWindow() + e.snd.offset[m];
        }
        auto off = static_cast<std::ptrdiff_t>(reinterpret_cast<std::uintptr_t>(base));
        for (auto const& cct : kv.second) {
            tags.push_back(CommSendBufTag<value_type>{
                .sfab = this->const_array(cct.srcIndex),
                .poff = off, .bx = cct.sbox});
            off += cct.sbox.numPts() * ncomp * sizeof(BUF);
        }
        ++m;
    }
    e.send_tv_shm.define(tags);
    e.shm_ready = true;
}

template <class FAB>
template <typename BUF>
void
FabArray<FAB>::FB_shm_nowait (const FB& TheFB, int scomp, int ncomp, bool sumboundary)
{
    auto* e = FB_on_arrival_meta<BUF>(TheFB, ncomp);

    const int N_locs = TheFB.m_LocTags->size();
    const auto N_rcvs = static_cast<int>(e->rcv.rank.size());
    const auto N_snds = static_cast<int>(e->snd.rank.size());

    fbd = std::make_unique<FBData<FAB>>();
    fbd->fb    = &TheFB;
    fbd->scomp = scomp;
    fbd->ncomp = ncomp;
    fbd->tag   = 0;
    fbd->oa_meta = e;
    fbd->shm_mode = static_cast<int>(NVSHMEM::fbMode());

    // All exchanges of a rank share its windows.
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(NVSHMEM::inFlight() == 0,
        "NVSHMEM FillBoundary supports one FillBoundary in flight at a time");
    ++NVSHMEM::inFlight();

    if (!e->shm_ready) { FB_shm_meta<BUF>(TheFB, ncomp, *e); }

    // Number this exchange for every rank pair.
    const int me = ParallelDescriptor::MyProc();
    detail::FBShmPeers rcv{}, snd{}, direct{};
    detail::FBShmPuts puts{};
    for (int i = 0; i < N_rcvs; ++i) {
        if (e->rcv.size[i] > 0) {
            const int r = e->rcv.rank[i];
            const auto n = NVSHMEM::nextExpected(r);
            rcv.add(r, n);
            fbd->shm_expected.push_back(n);
        }
    }
    for (int j = 0; j < N_snds; ++j) {
        if (e->snd.size[j] > 0) {
            const int d = e->snd.rank[j];
            const auto n = NVSHMEM::nextSent(d);
            snd.add(d, n);
            if (e->shm_is_put[j]) {
                puts.add(d, n, e->snd.offset[j], e->shm_roff[j], e->snd.size[j]);
            } else {
                direct.add(d, n);
            }
        }
    }

    {
        BL_PROFILE("FB::shm_post");
        detail::fb_shm_ready(rcv, snd, NVSHMEM::readyFlags(), me);
        if (N_snds > 0) {
            detail::ParallelFor_doit(e->send_tv_shm,
                [=] AMREX_GPU_DEVICE (
                    int icell, int ncells, int i, int j, int k, CommSendBufTag<value_type> const& tag) noexcept
            {
                if (icell < ncells) {
                    Array4<BUF> dfab{reinterpret_cast<BUF*>(static_cast<std::uintptr_t>(tag.poff)),
                                     amrex::begin(tag.bx), amrex::end(tag.bx), ncomp};
                    for (int n = 0; n < ncomp; ++n) {
                        dfab(i,j,k,n) = (BUF)tag.sfab(i,j,k,n+scomp);
                    }
                    // make the stores visible to other GPUs before the done flag
                    __threadfence_system();
                }
            });
            detail::fb_shm_done(direct, NVSHMEM::doneFlags(), me);
            detail::fb_shm_put(puts, NVSHMEM::recvWindow(), NVSHMEM::sendWindow(), NVSHMEM::doneFlags(), me);
        }
    }

    if (N_locs > 0) {
        if (sumboundary) {
            if constexpr (amrex::IsAddAssignable<value_type>::value) {
                BL_PROFILE("FB::local_add");
                FB_local_add_gpu(TheFB, scomp, ncomp, false);
            }
        } else {
            BL_PROFILE("FB::local_copy");
            FB_local_copy_gpu(TheFB, scomp, ncomp);
        }
    }
}

template <class FAB>
template <typename BUF>
void
FabArray<FAB>::FB_shm_finish ()
{
    auto* e = fbd->oa_meta;
    const FB* TheFB = fbd->fb;
    const auto N_rcvs = static_cast<int>(e->rcv.rank.size());

    --NVSHMEM::inFlight();

    if (N_rcvs > 0) {
        detail::FBShmPeers rcv{};
        int m = 0;
        for (int i = 0; i < N_rcvs; ++i) {
            if (e->rcv.size[i] > 0) { rcv.add(e->rcv.rank[i], fbd->shm_expected[m++]); }
        }
        detail::fb_shm_wait(rcv, NVSHMEM::doneFlags());

        BL_PROFILE("FB::unpack");
        char const* rbase = NVSHMEM::recvWindow();
        const bool sumboundary = TheFB->m_sb_snghost.allGE(0);
        using T = value_type;
        if (sumboundary) {
            if constexpr (amrex::IsAddAssignable<T>::value) {
                if (TheFB->m_threadsafe_rcv) {
                    detail::ParallelFor_doit(e->recv_tv, detail::FBOnArrivalUnpack<BUF,T,true,false>
                                             {rbase, fbd->scomp, fbd->ncomp});
                } else if constexpr (amrex::HasAtomicAdd<T>::value) {
                    detail::ParallelFor_doit(e->recv_tv, detail::FBOnArrivalUnpack<BUF,T,true,true>
                                             {rbase, fbd->scomp, fbd->ncomp});
                } else {
                    amrex::Abort("FB_shm_finish: SumBoundary needs thread-safe tags or atomic add");
                }
            }
        } else {
            detail::ParallelFor_doit(e->recv_tv, detail::FBOnArrivalUnpack<BUF,T,false,false>
                                     {rbase, fbd->scomp, fbd->ncomp});
        }
    }

    if (!Gpu::inNoSyncRegion()) {
        BL_PROFILE("FB::unpack_sync");
        Gpu::streamSynchronize();
    }

    fbd.reset();
}

}

#endif
#endif
