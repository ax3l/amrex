#
# Kokkos: on-node performance-portability layer
#
# AMReX uses Kokkos for kernel launches, reductions, atomics, memory
# spaces and execution-space (stream) handling.  The Kokkos installation
# must be configured consistently with AMReX:
#
#   AMReX_GPU_BACKEND=CUDA  <->  Kokkos_ENABLE_CUDA (compiled as CMake CUDA language)
#   AMReX_GPU_BACKEND=HIP   <->  Kokkos_ENABLE_HIP
#   AMReX_GPU_BACKEND=SYCL  <->  Kokkos_ENABLE_SYCL
#   AMReX_GPU_BACKEND=NONE  <->  no Kokkos device backend
#
# Kokkos_ENABLE_SERIAL is always required, and Kokkos_ENABLE_OPENMP is
# required if AMReX_OMP=ON.
#
# Kokkos is searched with find_package (e.g., set Kokkos_ROOT).  If it is
# not found, or if AMReX_kokkos_internal=ON, AMReX downloads and builds
# Kokkos with a matching configuration (AMReX_kokkos_src, AMReX_kokkos_repo
# and AMReX_kokkos_branch control the source).
#
if (NOT AMReX_KOKKOS)
   return()
endif ()

# also used by the exported AMReXConfig.cmake
set(AMReX_KOKKOS_MIN_VERSION 5.0 CACHE INTERNAL "Minimum Kokkos version")

set(AMReX_kokkos_src "" CACHE PATH
    "Local path to Kokkos source directory (preferred if set)")
set(AMReX_kokkos_repo "https://github.com/kokkos/kokkos.git" CACHE STRING
    "Repository URI to pull and build Kokkos from if(AMReX_kokkos_internal)")
set(AMReX_kokkos_branch "5.3.0" CACHE STRING
    "Repository branch for AMReX_kokkos_repo if(AMReX_kokkos_internal)")

# AMReX targets are compiled without compiler-specific C++ extensions
# (see AMReX_Config.cmake); Kokkos requires the same.
if (NOT DEFINED CMAKE_CXX_EXTENSIONS)
   set(CMAKE_CXX_EXTENSIONS OFF)
endif ()

if (NOT TARGET Kokkos::kokkos)
   if (NOT AMReX_kokkos_internal AND NOT AMReX_kokkos_src)
      find_package(Kokkos ${AMReX_KOKKOS_MIN_VERSION} CONFIG QUIET)
   endif ()
   if (NOT Kokkos_FOUND)
      if (DEFINED AMReX_kokkos_internal AND NOT AMReX_kokkos_internal AND NOT AMReX_kokkos_src)
         message(FATAL_ERROR
            "AMReX: Kokkos >= ${AMReX_KOKKOS_MIN_VERSION} not found. Set Kokkos_ROOT, "
            "or let AMReX build Kokkos with -DAMReX_kokkos_internal=ON.")
      endif ()
      include(AMReXKokkosInternal)
   endif ()
endif ()

if (Kokkos_FOUND)
   message(STATUS "Kokkos found: ${Kokkos_DIR} (version ${Kokkos_VERSION})")
endif ()

# Backend consistency checks
function (amrex_kokkos_require _backend _required _reason)
   if (TARGET kokkoscore)
      # Kokkos is built as part of this project
      set(_has ${Kokkos_ENABLE_${_backend}})
   else ()
      kokkos_check(DEVICES ${_backend} RETURN_VALUE _has)
   endif ()
   if (_required AND NOT _has)
      message(FATAL_ERROR "AMReX: Kokkos must be built with Kokkos_ENABLE_${_backend}=ON (${_reason}).")
   elseif (NOT _required AND _has)
      message(FATAL_ERROR "AMReX: Kokkos must not be built with Kokkos_ENABLE_${_backend}=ON (${_reason}).")
   endif ()
endfunction ()

set(_amrex_kokkos_cuda OFF)
set(_amrex_kokkos_hip OFF)
set(_amrex_kokkos_sycl OFF)
if (AMReX_GPU_BACKEND STREQUAL "CUDA")
   set(_amrex_kokkos_cuda ON)
elseif (AMReX_GPU_BACKEND STREQUAL "HIP")
   set(_amrex_kokkos_hip ON)
elseif (AMReX_GPU_BACKEND STREQUAL "SYCL")
   set(_amrex_kokkos_sycl ON)
endif ()

amrex_kokkos_require(CUDA ${_amrex_kokkos_cuda} "must match AMReX_GPU_BACKEND=${AMReX_GPU_BACKEND}")
amrex_kokkos_require(HIP  ${_amrex_kokkos_hip}  "must match AMReX_GPU_BACKEND=${AMReX_GPU_BACKEND}")
amrex_kokkos_require(SYCL ${_amrex_kokkos_sycl} "must match AMReX_GPU_BACKEND=${AMReX_GPU_BACKEND}")
amrex_kokkos_require(SERIAL ON "AMReX host code runs on Kokkos::Serial")
if (AMReX_OMP)
   amrex_kokkos_require(OPENMP ON "AMReX_OMP=ON")
endif ()

# Kokkos' atomics require the same relocatable device code mode in Kokkos and AMReX.
if ((AMReX_GPU_BACKEND STREQUAL "CUDA" OR AMReX_GPU_BACKEND STREQUAL "HIP") AND NOT TARGET kokkoscore)
   kokkos_check(OPTIONS ${AMReX_GPU_BACKEND}_RELOCATABLE_DEVICE_CODE RETURN_VALUE _kokkos_rdc)
   if (_kokkos_rdc AND NOT AMReX_GPU_RDC)
      message(FATAL_ERROR "AMReX: Kokkos was built with Kokkos_ENABLE_${AMReX_GPU_BACKEND}_RELOCATABLE_DEVICE_CODE=ON; "
                          "build AMReX with AMReX_GPU_RDC=ON or use a Kokkos built without it.")
   elseif (NOT _kokkos_rdc AND AMReX_GPU_RDC)
      message(FATAL_ERROR "AMReX: AMReX_GPU_RDC=ON requires a Kokkos built with "
                          "Kokkos_ENABLE_${AMReX_GPU_BACKEND}_RELOCATABLE_DEVICE_CODE=ON; "
                          "build AMReX with AMReX_GPU_RDC=OFF, use such a Kokkos, or let AMReX "
                          "build a matching Kokkos with AMReX_kokkos_internal=ON.")
   endif ()
endif ()

if (AMReX_GPU_BACKEND STREQUAL "CUDA")
   # AMReX compiles its sources as the CMake CUDA language; a Kokkos that
   # expects nvcc_wrapper as the C++ compiler cannot be mixed with that.
   if (NOT TARGET kokkoscore AND NOT Kokkos_COMPILE_LANGUAGE STREQUAL "CUDA")
      message(FATAL_ERROR
         "AMReX: with AMReX_GPU_BACKEND=CUDA, Kokkos must be built with "
         "Kokkos_ENABLE_COMPILE_AS_CMAKE_LANGUAGE=ON (found Kokkos_COMPILE_LANGUAGE="
         "'${Kokkos_COMPILE_LANGUAGE}').")
   endif ()
endif ()

foreach(D IN LISTS AMReX_SPACEDIM)
   target_link_libraries(amrex_${D}d PUBLIC Kokkos::kokkos)
endforeach()
