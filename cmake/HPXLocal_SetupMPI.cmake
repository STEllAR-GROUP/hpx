# Copyright (c) 2019 Ste||ar Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

# FIXME : in the future put it directly inside the cmake directory of the
# corresponding plugin

include(HPXLocal_Message)

macro(hpx_local_setup_mpi)
  if(NOT TARGET Mpi::mpi)
    find_package(MPI)
    # cmake version don't have the same found variable set
    if(NOT MPI_FOUND AND NOT MPI_CXX_FOUND)
      hpx_local_error(
        "MPI could not be found but was requested by your configuration, \n"
        "please specify MPI_ROOT to point to the root of your MPI installation"
      )
    endif()
    add_library(Mpi::mpi INTERFACE IMPORTED)
    target_include_directories(
      Mpi::mpi SYSTEM INTERFACE ${MPI_INCLUDE_PATH} ${MPI_CXX_INCLUDE_DIRS}
    )
    # MPI_LIBRARY and EXTRA is deprecated but still linked for older MPI
    # versions
    if(MPI_CXX_LIBRARIES)
      target_link_libraries(Mpi::mpi INTERFACE ${MPI_CXX_LIBRARIES})
    endif()
    # Ensure compatibility with older versions
    if(MPI_LIBRARY)
      target_link_libraries(Mpi::mpi INTERFACE ${MPI_LIBRARY})
    endif()
    if(MPI_EXTRA_LIBRARY)
      target_link_libraries(Mpi::mpi INTERFACE ${MPI_EXTRA_LIBRARY})
    endif()
    separate_arguments(MPI_CXX_COMPILE_FLAGS)
    target_compile_options(Mpi::mpi INTERFACE ${MPI_CXX_COMPILE_FLAGS})
    target_compile_definitions(
      Mpi::mpi INTERFACE ${MPI_CXX_COMPILE_DEFINITIONS}
    )

    hpx_local_info("MPI version: " ${MPI_C_VERSION})
  endif()
endmacro()

if(HPXLocal_WITH_ASYNC_MPI)
  hpx_local_setup_mpi()
endif()
