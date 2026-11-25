#include <AMReX_NCCL.H>

#ifdef AMREX_USE_NCCL

#include <AMReX.H>
#include <AMReX_GpuDevice.H>
#include <AMReX_GpuError.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Print.H>
#include <AMReX_Vector.H>

#include <nccl_device.h>

#include <string>

namespace amrex::NCCL {

namespace {
    FBMode s_fb_mode = FBMode::off;
    ncclComm_t s_comm = nullptr;

    constexpr std::size_t s_flag_bytes = std::size_t(2) << 20;
    std::size_t s_window_size = 0;
    void* s_window = nullptr;
    void* s_flags = nullptr;
    ncclWindow_t s_win = nullptr;
    ncclWindow_t s_flag_win = nullptr;
    int s_nprocs = 0;
    Vector<char*> s_peer_window;
    Vector<unsigned long long*> s_peer_done;
    Vector<unsigned long long*> s_peer_ready;
    Vector<unsigned long long> s_sent;
    Vector<unsigned long long> s_expected;
    bool s_fence = true;
    int s_lsa_in_flight = 0;
}

void check (ncclResult_t r, const char* what)
{
    if (r != ncclSuccess) {
        amrex::Abort(std::string("NCCL error in ") + what + ": " + ncclGetErrorString(r));
    }
}

void Initialize ()
{
    int mode = 0;
    Long window_mb = 256;
    {
        ParmParse pp("amrex.nccl");
        pp.query("fb_mode", mode);
        pp.query("window_mb", window_mb);
        pp.query("fence", s_fence);
    }

    s_fb_mode = FBMode::off;
    s_nprocs = ParallelDescriptor::NProcs();
    if (mode == 0 || s_nprocs == 1) { return; }
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(mode == 1 || mode == 2, "amrex.nccl.fb_mode must be 0, 1 or 2");

    const int myproc = ParallelDescriptor::MyProc();
    MPI_Comm mpi_comm = ParallelDescriptor::Communicator();

    ncclUniqueId id;
    if (myproc == 0) { check(ncclGetUniqueId(&id), "ncclGetUniqueId"); }
    MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, mpi_comm);

    ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
    config.numRmaCtx = 0; // no host one-sided operations
    check(ncclCommInitRankConfig(&s_comm, s_nprocs, id, myproc, &config), "ncclCommInitRankConfig");
    s_fb_mode = FBMode::sendrecv;

    if (mode == 2) {
        ncclCommProperties_t props = NCCL_COMM_PROPERTIES_INITIALIZER;
        check(ncclCommQueryProperties(s_comm, &props), "ncclCommQueryProperties");
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(props.deviceApiSupport && props.nLsaTeams == 1,
            "amrex.nccl.fb_mode=2 needs all ranks in one NVLink load/store domain");
        AMREX_ALWAYS_ASSERT(2*std::size_t(s_nprocs)*sizeof(unsigned long long) <= s_flag_bytes);

        s_window_size = std::size_t(window_mb) << 20;
        check(ncclMemAlloc(&s_window, s_window_size), "ncclMemAlloc");
        check(ncclMemAlloc(&s_flags, s_flag_bytes), "ncclMemAlloc");
        AMREX_CUDA_SAFE_CALL(cudaMemset(s_flags, 0, s_flag_bytes));
        check(ncclCommWindowRegister(s_comm, s_window, s_window_size, &s_win, NCCL_WIN_COLL_SYMMETRIC),
              "ncclCommWindowRegister");
        check(ncclCommWindowRegister(s_comm, s_flags, s_flag_bytes, &s_flag_win, NCCL_WIN_COLL_SYMMETRIC),
              "ncclCommWindowRegister");
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(s_win && s_flag_win, "NCCL symmetric memory windows are not available");

        s_peer_window.assign(s_nprocs, nullptr);
        s_peer_done.assign(s_nprocs, nullptr);
        s_peer_ready.assign(s_nprocs, nullptr);
        s_sent.assign(s_nprocs, 0);
        s_expected.assign(s_nprocs, 0);
        const ncclTeam_t world = ncclTeamWorld(s_comm);
        for (int r = 0; r < s_nprocs; ++r) {
            if (r == myproc) { continue; }
            const int lsa_rank = ncclTeamRankToLsa(s_comm, world, r);
            void* p = nullptr;
            check(ncclGetLsaDevicePointer(s_win, 0, lsa_rank, &p), "ncclGetLsaDevicePointer");
            s_peer_window[r] = static_cast<char*>(p);
            check(ncclGetLsaDevicePointer(s_flag_win, std::size_t(myproc)*sizeof(unsigned long long),
                                          lsa_rank, &p), "ncclGetLsaDevicePointer");
            s_peer_done[r] = static_cast<unsigned long long*>(p);
            check(ncclGetLsaDevicePointer(s_flag_win, std::size_t(s_nprocs+myproc)*sizeof(unsigned long long),
                                          lsa_rank, &p), "ncclGetLsaDevicePointer");
            s_peer_ready[r] = static_cast<unsigned long long*>(p);
        }
        s_fb_mode = FBMode::lsa;
    }

    if (amrex::Verbose() > 0) {
        amrex::Print() << "NCCL initialized, FillBoundary mode " << mode << "\n";
    }
}

void Finalize ()
{
    if (s_comm == nullptr) { return; }
    if (s_win) { check(ncclCommWindowDeregister(s_comm, s_win), "ncclCommWindowDeregister"); }
    if (s_flag_win) { check(ncclCommWindowDeregister(s_comm, s_flag_win), "ncclCommWindowDeregister"); }
    if (s_window) { check(ncclMemFree(s_window), "ncclMemFree"); }
    if (s_flags) { check(ncclMemFree(s_flags), "ncclMemFree"); }
    check(ncclCommFinalize(s_comm), "ncclCommFinalize");
    check(ncclCommDestroy(s_comm), "ncclCommDestroy");
    s_comm = nullptr;
    s_win = nullptr;
    s_flag_win = nullptr;
    s_window = nullptr;
    s_flags = nullptr;
    s_fb_mode = FBMode::off;
}

FBMode fbMode () noexcept { return s_fb_mode; }
ncclComm_t comm () noexcept { return s_comm; }

char* recvWindow () noexcept { return static_cast<char*>(s_window); }
std::size_t recvWindowSize () noexcept { return s_window_size; }
char* peerRecvWindow (int rank) noexcept { return s_peer_window[rank]; }

const unsigned long long* doneFlags () noexcept
{
    return static_cast<const unsigned long long*>(s_flags);
}

const unsigned long long* readyFlags () noexcept
{
    return static_cast<const unsigned long long*>(s_flags) + s_nprocs;
}

unsigned long long* peerDoneFlag (int rank) noexcept { return s_peer_done[rank]; }
unsigned long long* peerReadyFlag (int rank) noexcept { return s_peer_ready[rank]; }
unsigned long long nextSent (int rank) noexcept { return ++s_sent[rank]; }
unsigned long long nextExpected (int rank) noexcept { return ++s_expected[rank]; }
int& lsaInFlight () noexcept { return s_lsa_in_flight; }
bool fence () noexcept { return s_fence; }

}

#endif
