#
# Download (or take from AMReX_kokkos_src) and build Kokkos as part of AMReX,
# configured to match AMReX's GPU backend, GPU architecture, OpenMP and
# relocatable device code settings.
#
include(FetchContent)

set(Kokkos_ENABLE_SERIAL ON CACHE BOOL "" FORCE)
set(Kokkos_ENABLE_OPENMP ${AMReX_OMP} CACHE BOOL "" FORCE)
set(Kokkos_ENABLE_TESTS OFF CACHE BOOL "")
set(Kokkos_ENABLE_EXAMPLES OFF CACHE BOOL "")
set(Kokkos_ENABLE_BENCHMARKS OFF CACHE BOOL "")

if (AMReX_GPU_BACKEND STREQUAL "CUDA")
   set(Kokkos_ENABLE_CUDA ON CACHE BOOL "" FORCE)
   # AMReX compiles its sources as the CMake CUDA language
   set(Kokkos_ENABLE_COMPILE_AS_CMAKE_LANGUAGE ON CACHE BOOL "" FORCE)
   set(Kokkos_ENABLE_CUDA_RELOCATABLE_DEVICE_CODE ${AMReX_GPU_RDC} CACHE BOOL "" FORCE)

   # Kokkos supports one GPU architecture.  Without an explicit one (e.g.,
   # CMAKE_CUDA_ARCHITECTURES=native), Kokkos detects the GPU of this machine.
   set(_amrex_kokkos_cuda_archs "${AMREX_CUDA_ARCHS}")
   list(TRANSFORM _amrex_kokkos_cuda_archs REPLACE "-real|-virtual" "")
   list(REMOVE_DUPLICATES _amrex_kokkos_cuda_archs)
   list(FILTER _amrex_kokkos_cuda_archs INCLUDE REGEX "^[0-9]+$")
   list(LENGTH _amrex_kokkos_cuda_archs _amrex_kokkos_ncuda_archs)
   if (_amrex_kokkos_ncuda_archs GREATER 1)
      list(SORT _amrex_kokkos_cuda_archs COMPARE NATURAL)
      list(GET _amrex_kokkos_cuda_archs 0 _amrex_kokkos_cuda_arch)
      message(WARNING "AMReX: Kokkos supports a single CUDA architecture; "
                      "building Kokkos for ${_amrex_kokkos_cuda_arch} out of ${AMREX_CUDA_ARCHS}.")
   elseif (_amrex_kokkos_ncuda_archs EQUAL 1)
      set(_amrex_kokkos_cuda_arch ${_amrex_kokkos_cuda_archs})
   endif ()
   if (DEFINED _amrex_kokkos_cuda_arch)
      set(_amrex_kokkos_cuda_arch_map
          60 PASCAL60 61 PASCAL61 70 VOLTA70 72 VOLTA72 75 TURING75 80 AMPERE80
          86 AMPERE86 87 AMPERE87 89 ADA89 90 HOPPER90 100 BLACKWELL100
          103 BLACKWELL103 107 RUBIN107 120 BLACKWELL120 121 BLACKWELL121)
      list(FIND _amrex_kokkos_cuda_arch_map ${_amrex_kokkos_cuda_arch} _idx)
      if (_idx LESS 0)
         message(FATAL_ERROR "AMReX: no Kokkos architecture for CUDA compute capability ${_amrex_kokkos_cuda_arch}")
      endif ()
      math(EXPR _idx "${_idx}+1")
      list(GET _amrex_kokkos_cuda_arch_map ${_idx} _amrex_kokkos_arch)
      set(Kokkos_ARCH_${_amrex_kokkos_arch} ON CACHE BOOL "" FORCE)
   endif ()
elseif (AMReX_GPU_BACKEND STREQUAL "HIP")
   set(Kokkos_ENABLE_HIP ON CACHE BOOL "" FORCE)
   set(Kokkos_ENABLE_HIP_RELOCATABLE_DEVICE_CODE ${AMReX_GPU_RDC} CACHE BOOL "" FORCE)
   list(GET AMReX_AMD_ARCH 0 _amrex_kokkos_amd_arch)
   string(TOUPPER "${_amrex_kokkos_amd_arch}" _amrex_kokkos_amd_arch)
   string(REGEX REPLACE ":.*" "" _amrex_kokkos_amd_arch "${_amrex_kokkos_amd_arch}")
   set(Kokkos_ARCH_AMD_${_amrex_kokkos_amd_arch} ON CACHE BOOL "" FORCE)
elseif (AMReX_GPU_BACKEND STREQUAL "SYCL")
   set(Kokkos_ENABLE_SYCL ON CACHE BOOL "" FORCE)
   if (AMReX_INTEL_ARCH AND NOT AMReX_INTEL_ARCH STREQUAL "IGNORE")
      string(TOUPPER "${AMReX_INTEL_ARCH}" _amrex_kokkos_intel_arch)
      set(Kokkos_ARCH_INTEL_${_amrex_kokkos_intel_arch} ON CACHE BOOL "" FORCE)
   endif ()
endif ()

if (AMReX_kokkos_src)
   message(STATUS "Compiling local Kokkos ...")
   message(STATUS "Kokkos source path: ${AMReX_kokkos_src}")
   if (NOT IS_DIRECTORY ${AMReX_kokkos_src})
      message(FATAL_ERROR "Specified directory AMReX_kokkos_src='${AMReX_kokkos_src}' does not exist!")
   endif ()
   add_subdirectory(${AMReX_kokkos_src} _deps/localkokkos-build/)
else ()
   message(STATUS "Downloading Kokkos ...")
   message(STATUS "Kokkos repository: ${AMReX_kokkos_repo} (${AMReX_kokkos_branch})")
   FetchContent_Declare(fetchedkokkos
       GIT_REPOSITORY ${AMReX_kokkos_repo}
       GIT_TAG        ${AMReX_kokkos_branch}
       GIT_SHALLOW    ON
       BUILD_IN_SOURCE 0
   )
   FetchContent_MakeAvailable(fetchedkokkos)
endif ()

set(Kokkos_FOUND TRUE)
