# Copyright (c) 2019 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

cmake_minimum_required(VERSION 3.3.2 FATAL_ERROR)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

find_package(MPI)

if(${CMAKE_VERSION} VERSION_LESS "3.12.0")
    message(ERROR "cmake 3.12 or newer required")
endif()

# All cmake version don't have the same found variable set
if(NOT MPI_FOUND AND NOT MPI_C_FOUND)
  hpx_error("MPI could not be found and hpx::mpi library was requested\n
  MPI_ROOT to point to the root of your MPI installation")
endif()

if (NOT TARGET hpx::mpi )
  add_library(hpx::mpi INTERFACE IMPORTED)
  set_property(TARGET hpx::mpi PROPERTY
    INTERFACE_INCLUDE_DIRECTORIES ${MPI_INCLUDE_PATH} ${MPI_C_INCLUDE_DIRS})

  if (MPI_C_LIBRARIES)
      target_link_libraries(hpx::mpi INTERFACE ${MPI_CXX_LIBRARY})
  endif()
  # Ensure compatibility with older versions
  if (MPI_LIBRARY)
    target_link_libraries(hpx::mpi INTERFACE ${MPI_LIBRARY})
  endif()
  if (MPI_EXTRA_LIBRARY)
    target_link_libraries(hpx::mpi INTERFACE ${MPI_EXTRA_LIBRARY})
  endif()

  set_property(TARGET hpx::mpi PROPERTY INTERFACE_COMPILE_OPTIONS ${MPI_CXX_COMPILE_FLAGS})
    set_property(TARGET hpx::mpi PROPERTY INTERFACE_COMPILE_DEFINITIONS ${MPI_CXX_COMPILE_DEFINITIONS})
endif()

# Default location is $HPX_ROOT/libs/mpi/include
set(mpi_headers
    hpx/mpi/mpi_future.hpp
)

# Default location is $HPX_ROOT/libs/mpi/include_compatibility
set(mpi_compat_headers)

set(mpi_sources)

include(HPX_AddModule)
add_hpx_module(mpi
  COMPATIBILITY_HEADERS OFF
  DEPRECATION_WARNINGS
  FORCE_LINKING_GEN
  GLOBAL_HEADER_GEN OFF
  SOURCES ${mpi_sources}
  HEADERS ${mpi_headers}
  COMPAT_HEADERS ${mpi_compat_headers}
  DEPENDENCIES
    hpx_config
    hpx_assertion
    hpx_local_lcos
    hpx::mpi
  CMAKE_SUBDIRS examples tests
)

include(HPX_PrintSummary)
create_configuration_summary(
  "  Module configuration summary (mpi):" "mpi")
