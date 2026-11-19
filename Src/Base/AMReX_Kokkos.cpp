#include <AMReX_Kokkos.H>
#include <AMReX_GpuDevice.H>

#ifdef AMREX_USE_KOKKOS

#include <AMReX.H>
#include <AMReX_OpenMP.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Print.H>

#include <cstdlib>
#include <string>

namespace amrex::kokkos {

namespace {
    bool s_initialized_by_amrex = false;
    bool s_atexit_registered = false;

    void finalize_at_exit ()
    {
        if (Kokkos::is_initialized() && !Kokkos::is_finalized()) {
            Kokkos::finalize();
        }
    }
}

void
Initialize (int device_id)
{
    if (Kokkos::is_finalized()) {
        amrex::Abort("amrex::Initialize: Kokkos has already been finalized and cannot be re-initialized.");
    }

    if (Kokkos::is_initialized()) {
        // The application initialized Kokkos.  AMReX uses Kokkos' device.
#ifdef AMREX_USE_GPU
        if (device_id >= 0 && Kokkos::device_id() != device_id) {
            amrex::Abort("amrex::Initialize: Kokkos was initialized by the application on device "
                         + std::to_string(Kokkos::device_id()) + ", but AMReX selected device "
                         + std::to_string(device_id) + ". Pass Kokkos' device id to amrex::Initialize.");
        }
#endif
        if (amrex::Verbose()) {
            amrex::Print() << "Kokkos " << KOKKOS_VERSION_MAJOR << "." << KOKKOS_VERSION_MINOR << "."
                           << KOKKOS_VERSION_PATCH << " was initialized by the application\n";
        }
        return;
    }

    bool disable_warnings = true;
    {
        ParmParse pp("amrex");
        pp.queryAdd("kokkos_disable_warnings", disable_warnings);
    }

    Kokkos::InitializationSettings settings;
    settings.set_disable_warnings(disable_warnings);
#ifdef AMREX_USE_GPU
    if (device_id >= 0) {
        settings.set_device_id(device_id);
    }
#else
    amrex::ignore_unused(device_id);
#endif
#ifdef AMREX_USE_OMP
    settings.set_num_threads(OpenMP::get_max_threads());
#endif

    Kokkos::initialize(settings);
    s_initialized_by_amrex = true;

    // Kokkos cannot be re-initialized once it is finalized, but AMReX can
    // be initialized and finalized several times in a process.  AMReX
    // therefore finalizes the Kokkos instance it created at program exit.
    if (!s_atexit_registered) {
        std::atexit(finalize_at_exit);
        s_atexit_registered = true;
    }

    if (amrex::Verbose()) {
        amrex::Print() << "Kokkos " << KOKKOS_VERSION_MAJOR << "." << KOKKOS_VERSION_MINOR << "."
                       << KOKKOS_VERSION_PATCH << " initialized with execution space "
                       << Kokkos::DefaultExecutionSpace::name() << "\n";
    }
}

void
Finalize ()
{
    if (Kokkos::is_initialized() && !Kokkos::is_finalized()) {
        Kokkos::fence("amrex::Finalize");
    }
}

bool
InitializedByAMReX () noexcept
{
    return s_initialized_by_amrex;
}

#ifdef AMREX_USE_GPU

namespace {
    template <typename MemSpace>
    void* allocate_in (MemSpace const& space, std::size_t nbytes, char const* label,
                       std::string* error)
    {
        try {
            return space.allocate(label, nbytes);
        } catch (Kokkos::Experimental::BadAlloc const& ex) {
            if (error) { *error = ex.what(); }
            return nullptr;
        }
    }

#ifdef AMREX_USE_SYCL
    // Use AMReX's SYCL queue so that the memory belongs to AMReX's context.
    Kokkos::SYCLDeviceUSMSpace device_space () { return Kokkos::SYCLDeviceUSMSpace(Gpu::Device::streamQueue(0)); }
    Kokkos::SYCLSharedUSMSpace managed_space () { return Kokkos::SYCLSharedUSMSpace(Gpu::Device::streamQueue(0)); }
    Kokkos::SYCLHostUSMSpace pinned_space () { return Kokkos::SYCLHostUSMSpace(Gpu::Device::streamQueue(0)); }
#else
    Gpu::KokkosDeviceMemSpace device_space () { return Gpu::KokkosDeviceMemSpace{}; }
    Kokkos::SharedSpace managed_space () { return Kokkos::SharedSpace{}; }
    Kokkos::SharedHostPinnedSpace pinned_space () { return Kokkos::SharedHostPinnedSpace{}; }
#endif
}

void*
allocate (MemoryKind kind, std::size_t nbytes, char const* label, std::string* error)
{
    switch (kind) {
    case MemoryKind::Device:
        return allocate_in(device_space(), nbytes, label, error);
    case MemoryKind::Managed:
        return allocate_in(managed_space(), nbytes, label, error);
    case MemoryKind::Pinned:
        return allocate_in(pinned_space(), nbytes, label, error);
    }
    return nullptr;
}

void
deallocate (MemoryKind kind, void* p, std::size_t nbytes, char const* label)
{
    switch (kind) {
    case MemoryKind::Device:
        device_space().deallocate(label, p, nbytes);
        break;
    case MemoryKind::Managed:
        managed_space().deallocate(label, p, nbytes);
        break;
    case MemoryKind::Pinned:
        pinned_space().deallocate(label, p, nbytes);
        break;
    }
}

#endif

}

#endif
