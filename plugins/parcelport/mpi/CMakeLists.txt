# Copyright (c) 2007-2013 Hartmut Kaiser
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

include(HPX_AddLibrary)

################################################################################
# Decide whether to use the MPI based parcelport
################################################################################
if(HPX_PARCELPORT_MPI)
  find_package(MPI)
  if(NOT MPI_CXX_FOUND)
    hpx_error("MPI could not be found and HPX_PARCELPORT_MPI=On, please specify MPI_CXX_COMPILER to point to a working MPI C++ compiler for your platform")
  endif()
  hpx_add_config_define(HPX_PARCELPORT_MPI)
  if(MPI_CXX_COMPILE_FLAGS)
    hpx_add_compile_flag(${MPI_CXX_COMPILE_FLAGS})
  endif()
  if(MPI_CXX_LINK_FLAGS)
    #    hpx_add_link_flag_if_available(${MPI_CXX_LINK_FLAGS})
  endif()

  # This list is to detect whether we run inside an mpi environment.
  # If one of those environment variables is set, the MPI parcelport
  # is enabled by default.
  # PMI_RANK: Intel MPI and MVAPICH2
  # OMPI_COMM_WORLD_SIZE: OpenMPI starting at version 1.3
  if(HPX_PARCELPORT_MPI_ENV)
    string(REPLACE ";" "," hpx_parcelport_mpi_env_ "${HPX_PARCELPORT_MPI_ENV}")
    hpx_add_config_define(HPX_PARCELPORT_MPI_ENV "\"${hpx_parcelport_mpi_env_}\"")
  endif()

  macro(add_parcelport_mpi_module)
    if(MPI_CXX_INCLUDE_PATH)
      include_directories(${MPI_CXX_INCLUDE_PATH})
    endif()
    hpx_debug("add_parcelport_mpi_module")
    add_parcelport(mpi
      STATIC
      SOURCES
        "${hpx_SOURCE_DIR}/plugins/parcelport/mpi/connection_handler_mpi.cpp"
        "${hpx_SOURCE_DIR}/plugins/parcelport/mpi/mpi_environment.cpp"
        "${hpx_SOURCE_DIR}/plugins/parcelport/mpi/parcelport_mpi.cpp"
      HEADERS
        "${hpx_SOURCE_DIR}/hpx/plugins/parcelport/mpi/connection_handler.hpp"
        "${hpx_SOURCE_DIR}/hpx/plugins/parcelport/mpi/header.hpp"
        "${hpx_SOURCE_DIR}/hpx/plugins/parcelport/mpi/locality.hpp"
        "${hpx_SOURCE_DIR}/hpx/plugins/parcelport/mpi/mpi_environment.hpp"
        "${hpx_SOURCE_DIR}/hpx/plugins/parcelport/mpi/sender.hpp"
        "${hpx_SOURCE_DIR}/hpx/plugins/parcelport/mpi/receiver.hpp"
      DEPENDENCIES
        ${MPI_CXX_LIBRARIES}
        ${MPI_LIBRARY}
        ${MPI_EXTRA_LIBRARY}
      FOLDER "Core/Plugins/Parcelport/MPI"
    )
  endmacro()
else()
  macro(add_parcelport_mpi_module)
  endmacro()
endif()
