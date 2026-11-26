#include <AMReX_NVSHMEM.H>

#ifdef AMREX_USE_NVSHMEM

#include <AMReX.H>
#include <AMReX_GpuDevice.H>
#include <AMReX_GpuError.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Print.H>
#include <AMReX_Vector.H>

#include <nvshmem.h>
#include <nvshmemx.h>

namespace amrex::NVSHMEM {

namespace {
    FBMode s_fb_mode = FBMode::off;
    bool s_initialized = false;
    std::size_t s_window_size = 0;
    char* s_recv = nullptr;
    char* s_send = nullptr;
    std::uint64_t* s_flags = nullptr;
    int s_nprocs = 0;
    int s_in_flight = 0;
    Vector<char*> s_peer_recv;
    Vector<std::uint64_t> s_sent;
    Vector<std::uint64_t> s_expected;
}

void Initialize ()
{
    int mode = 0;
    Long window_mb = 256;
    {
        ParmParse pp("amrex.nvshmem");
        pp.query("fb_mode", mode);
        pp.query("window_mb", window_mb);
    }

    s_fb_mode = FBMode::off;
    s_nprocs = ParallelDescriptor::NProcs();
    if (mode == 0 || s_nprocs == 1) { return; }
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(mode == 1 || mode == 2, "amrex.nvshmem.fb_mode must be 0, 1 or 2");

    const int myproc = ParallelDescriptor::MyProc();
    MPI_Comm mpi_comm = ParallelDescriptor::Communicator();
    nvshmemx_init_attr_t attr = NVSHMEMX_INIT_ATTR_INITIALIZER;
    attr.mpi_comm = &mpi_comm;
    nvshmemx_init_attr(NVSHMEMX_INIT_WITH_MPI_COMM, &attr);
    s_initialized = true;
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(nvshmem_my_pe() == myproc && nvshmem_n_pes() == s_nprocs,
                                     "NVSHMEM PEs do not match the MPI ranks");

    s_window_size = std::size_t(window_mb) << 20;
    s_recv = static_cast<char*>(nvshmem_malloc(s_window_size));
    s_send = static_cast<char*>(nvshmem_malloc(s_window_size));
    const std::size_t flag_bytes = 2*std::size_t(s_nprocs)*sizeof(std::uint64_t);
    s_flags = static_cast<std::uint64_t*>(nvshmem_malloc(flag_bytes));
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(s_recv && s_send && s_flags,
        "nvshmem_malloc failed: raise NVSHMEM_SYMMETRIC_SIZE or lower amrex.nvshmem.window_mb");
    AMREX_CUDA_SAFE_CALL(cudaMemset(s_flags, 0, flag_bytes));
    ParallelDescriptor::Barrier(); // all flags are zero before anyone writes one

    s_peer_recv.assign(s_nprocs, nullptr);
    s_sent.assign(s_nprocs, 0);
    s_expected.assign(s_nprocs, 0);
    int nmapped = 0;
    if (mode == 2) {
        for (int r = 0; r < s_nprocs; ++r) {
            if (r == myproc) { continue; }
            s_peer_recv[r] = static_cast<char*>(nvshmem_ptr(s_recv, r));
            if (s_peer_recv[r]) { ++nmapped; }
        }
    }
    s_fb_mode = static_cast<FBMode>(mode);

    if (amrex::Verbose() > 0) {
        amrex::Print() << "NVSHMEM initialized, FillBoundary mode " << mode
                       << ", rank 0 maps " << nmapped << " of " << s_nprocs-1 << " peers\n";
    }
}

void Finalize ()
{
    if (!s_initialized) { return; }
    ParallelDescriptor::Barrier();
    nvshmem_free(s_recv);
    nvshmem_free(s_send);
    nvshmem_free(s_flags);
    nvshmem_finalize();
    s_recv = nullptr;
    s_send = nullptr;
    s_flags = nullptr;
    s_initialized = false;
    s_fb_mode = FBMode::off;
}

FBMode fbMode () noexcept { return s_fb_mode; }

char* recvWindow () noexcept { return s_recv; }
char* sendWindow () noexcept { return s_send; }
std::size_t windowSize () noexcept { return s_window_size; }
char* peerRecvWindow (int rank) noexcept { return s_peer_recv[rank]; }

std::uint64_t* doneFlags () noexcept { return s_flags; }
std::uint64_t* readyFlags () noexcept { return s_flags + s_nprocs; }
std::uint64_t nextSent (int rank) noexcept { return ++s_sent[rank]; }
std::uint64_t nextExpected (int rank) noexcept { return ++s_expected[rank]; }
int& inFlight () noexcept { return s_in_flight; }

}

#endif
