#ifndef AMREX_FB_NCCL_H_
#define AMREX_FB_NCCL_H_

/**
 * \file AMReX_FBNccl.H
 * \brief Experimental GPU FillBoundary and SumBoundary paths over NVIDIA NCCL
 *        instead of MPI, selected with `amrex.nccl.fb_mode` (see AMReX_NCCL.H).
 *
 * Both paths queue all their work on the current GPU stream and need no
 * stream synchronization between packing and communication:
 *  - mode 1 packs into a send buffer and exchanges it with `ncclSend` and
 *    `ncclRecv`;
 *  - mode 2 lets the pack kernel write straight into the receiving GPU's
 *    buffer over NVLink. Flags in peer memory replace the MPI handshake: a
 *    receiver announces that its buffer is free, a sender announces that its
 *    data is complete, and short kernels on the stream wait for these flags.
 *    One such FillBoundary can be in flight at a time.
 */

#if defined(AMREX_USE_NCCL) && defined(AMREX_FB_UNPACK_ON_ARRIVAL)

#include <AMReX_NCCL.H>

#include <cstdint>

namespace amrex {

/// \cond DOXYGEN_IGNORE
namespace detail {

constexpr int fb_nccl_max_peers = 32;

// Flags to write into peers: *p[m] = v[m].
struct FBNcclSet {
    unsigned long long* p[fb_nccl_max_peers];
    unsigned long long v[fb_nccl_max_peers];
    int n = 0;

    void add (unsigned long long* a_p, unsigned long long a_v) {
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(n < fb_nccl_max_peers, "NCCL FillBoundary: too many neighbor ranks");
        p[n] = a_p; v[n] = a_v; ++n;
    }
};

// Local flags to wait for: flags[rank[m]] >= v[m].
struct FBNcclWait {
    int rank[fb_nccl_max_peers];
    unsigned long long v[fb_nccl_max_peers];
    int n = 0;

    void add (int a_rank, unsigned long long a_v) {
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(n < fb_nccl_max_peers, "NCCL FillBoundary: too many neighbor ranks");
        rank[n] = a_rank; v[n] = a_v; ++n;
    }
};

// One short kernel on the current stream: write flags into the peers, then
// wait for flags the peers write here.
inline void fb_nccl_ctl (FBNcclSet const& set, FBNcclWait const& wait, const unsigned long long* flags)
{
    if (set.n == 0 && wait.n == 0) { return; }
    const volatile unsigned long long* f = flags;
    amrex::ParallelFor(1, [=] AMREX_GPU_DEVICE (int) noexcept
    {
        for (int m = 0; m < set.n; ++m) { *set.p[m] = set.v[m]; }
        __threadfence_system();
        for (int m = 0; m < wait.n; ++m) {
            while (f[wait.rank[m]] < wait.v[m]) {}
        }
    });
}

}
/// \endcond

// Mode 2: learn where this rank's messages go in the receivers' windows and
// build the pack tags that point there.
template <class FAB>
template <typename BUF>
void
FabArray<FAB>::FB_nccl_lsa_meta (const FB& TheFB, int ncomp, FBOnArrivalMeta<FAB>& e)
{
    BL_PROFILE("FB::nccl_lsa_meta");

    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(e.rcv.nbytes <= NCCL::recvWindowSize(),
        "NCCL FillBoundary: receive window too small, raise amrex.nccl.window_mb");

    const auto N_rcvs = static_cast<int>(e.rcv.rank.size());
    const auto N_snds = static_cast<int>(e.snd.rank.size());
    MPI_Comm comm = ParallelDescriptor::Communicator();
    const int tag = 27182; // one such message per rank pair and layout, in the same order on both sides

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

    // Same tag order as send_tv, but poff holds the address in the receiver.
    Vector<CommSendBufTag<value_type>> tags;
    int m = 0;
    for (auto const& kv : *TheFB.m_SndTags) {
        auto off = static_cast<std::ptrdiff_t>
            (reinterpret_cast<std::uintptr_t>(NCCL::peerRecvWindow(kv.first)) + peer_off[m]);
        for (auto const& cct : kv.second) {
            tags.push_back(CommSendBufTag<value_type>{
                .sfab = this->const_array(cct.srcIndex),
                .poff = off, .bx = cct.sbox});
            off += cct.sbox.numPts() * ncomp * sizeof(BUF);
        }
        ++m;
    }
    e.send_tv_lsa.define(tags);
    e.lsa_ready = true;
}

template <class FAB>
template <typename BUF>
void
FabArray<FAB>::FB_nccl_nowait (const FB& TheFB, int scomp, int ncomp, bool sumboundary)
{
    auto* e = FB_on_arrival_meta<BUF>(TheFB, ncomp);
    const NCCL::FBMode mode = NCCL::fbMode();

    const int N_locs = TheFB.m_LocTags->size();
    const auto N_rcvs = static_cast<int>(e->rcv.rank.size());
    const auto N_snds = static_cast<int>(e->snd.rank.size());

    fbd = std::make_unique<FBData<FAB>>();
    fbd->fb    = &TheFB;
    fbd->scomp = scomp;
    fbd->ncomp = ncomp;
    fbd->tag   = 0;
    fbd->oa_meta = e;
    fbd->nccl_mode = static_cast<int>(mode);

    if (mode == NCCL::FBMode::sendrecv)
    {
        auto alloc = [] (std::size_t n) {
            return n > 0 ? static_cast<char*>(The_Comms_Arena()->alloc(n)) : nullptr;
        };
        fbd->the_recv_data = alloc(e->rcv.nbytes);
        fbd->the_send_data = alloc(e->snd.nbytes);

        if (N_snds > 0) {
            BL_PROFILE("FB::pack");
            detail::fb_oa_pack<BUF>(e->send_tv, fbd->the_send_data, scomp, ncomp);
        }

        if (N_rcvs > 0 || N_snds > 0) {
            BL_PROFILE("FB::nccl_post");
            ncclComm_t comm = NCCL::comm();
            cudaStream_t stream = Gpu::gpuStream();
            NCCL::check(ncclGroupStart(), "ncclGroupStart");
            for (int i = 0; i < N_rcvs; ++i) {
                if (e->rcv.size[i] > 0) {
                    NCCL::check(ncclRecv(fbd->the_recv_data + e->rcv.offset[i], e->rcv.size[i],
                                         ncclChar, e->rcv.rank[i], comm, stream), "ncclRecv");
                }
            }
            for (int j = 0; j < N_snds; ++j) {
                if (e->snd.size[j] > 0) {
                    NCCL::check(ncclSend(fbd->the_send_data + e->snd.offset[j], e->snd.size[j],
                                         ncclChar, e->snd.rank[j], comm, stream), "ncclSend");
                }
            }
            NCCL::check(ncclGroupEnd(), "ncclGroupEnd");
        }
    }
    else
    {
        // All exchanges of a rank share its one receive window.
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(NCCL::lsaInFlight() == 0,
            "NCCL FillBoundary mode 2 supports one FillBoundary in flight at a time");
        ++NCCL::lsaInFlight();

        if (!e->lsa_ready) { FB_nccl_lsa_meta<BUF>(TheFB, ncomp, *e); }

        // Number this exchange for every rank pair.
        detail::FBNcclSet set_ready{}, set_done{};
        detail::FBNcclWait wait_ready{};
        for (int i = 0; i < N_rcvs; ++i) {
            if (e->rcv.size[i] > 0) {
                const int r = e->rcv.rank[i];
                const auto n = NCCL::nextExpected(r);
                fbd->nccl_expected.push_back(n);
                set_ready.add(NCCL::peerReadyFlag(r), n);
            }
        }
        for (int j = 0; j < N_snds; ++j) {
            if (e->snd.size[j] > 0) {
                const int d = e->snd.rank[j];
                const auto n = NCCL::nextSent(d);
                wait_ready.add(d, n);
                set_done.add(NCCL::peerDoneFlag(d), n);
            }
        }

        BL_PROFILE("FB::pack_peer");
        // This rank's window is free: everything queued before is done by the
        // time the kernel runs. Then wait until the receivers' windows are.
        detail::fb_nccl_ctl(set_ready, wait_ready, NCCL::readyFlags());
        if (N_snds > 0) {
            const bool fence = NCCL::fence();
            detail::ParallelFor_doit(e->send_tv_lsa,
                [=] AMREX_GPU_DEVICE (
                    int icell, int ncells, int i, int j, int k, CommSendBufTag<value_type> const& tag) noexcept
            {
                if (icell < ncells) {
                    Array4<BUF> dfab{reinterpret_cast<BUF*>(static_cast<std::uintptr_t>(tag.poff)),
                                     amrex::begin(tag.bx), amrex::end(tag.bx), ncomp};
                    for (int n = 0; n < ncomp; ++n) {
                        dfab(i,j,k,n) = (BUF)tag.sfab(i,j,k,n+scomp);
                    }
                    // make the stores visible to the receiving GPU before the done flag
                    if (fence) { __threadfence_system(); }
                }
            });
            detail::fb_nccl_ctl(set_done, detail::FBNcclWait{}, nullptr);
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
FabArray<FAB>::FB_nccl_finish ()
{
    auto* e = fbd->oa_meta;
    const FB* TheFB = fbd->fb;
    const auto mode = static_cast<NCCL::FBMode>(fbd->nccl_mode);
    const auto N_rcvs = static_cast<int>(e->rcv.rank.size());

    if (mode == NCCL::FBMode::lsa) { --NCCL::lsaInFlight(); }

    if (N_rcvs > 0) {
        char const* rbase = fbd->the_recv_data;
        if (mode == NCCL::FBMode::lsa) {
            rbase = NCCL::recvWindow();
            detail::FBNcclWait wait_done{};
            int m = 0;
            for (int i = 0; i < N_rcvs; ++i) {
                if (e->rcv.size[i] > 0) {
                    wait_done.add(e->rcv.rank[i], fbd->nccl_expected[m++]);
                }
            }
            detail::fb_nccl_ctl(detail::FBNcclSet{}, wait_done, NCCL::doneFlags());
        }

        BL_PROFILE("FB::unpack");
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
                    amrex::Abort("FB_nccl_finish: SumBoundary needs thread-safe tags or atomic add");
                }
            }
        } else {
            detail::ParallelFor_doit(e->recv_tv, detail::FBOnArrivalUnpack<BUF,T,false,false>
                                     {rbase, fbd->scomp, fbd->ncomp});
        }
    }

    // In a no-sync region the buffers are released in stream order.
    const bool sync = !Gpu::inNoSyncRegion();
    if (sync) {
        BL_PROFILE("FB::unpack_sync");
        Gpu::streamSynchronize();
    }
    for (char* p : {fbd->the_recv_data, fbd->the_send_data}) {
        if (p) {
            if (sync) {
                The_Comms_Arena()->free(p);
            } else {
                Gpu::freeAsync(The_Comms_Arena(), p);
            }
        }
    }

    fbd.reset();
}

}

#endif
#endif
