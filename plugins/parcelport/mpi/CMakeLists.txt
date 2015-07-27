# Copyright (c) 2007-2013 Hartmut Kaiser
# Copyright (c) 2014-2015 Thomas Heller
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

include(HPX_AddLibrary)

################################################################################
# Decide whether to use the MPI based parcelport
################################################################################
if(HPX_WITH_PARCELPORT_MPI)
  find_package(MPI)
  if(NOT MPI_CXX_FOUND)
    hpx_error("MPI could not be found and HPX_WITH_PARCELPORT_MPI=On, please specify MPI_CXX_COMPILER to point to a working MPI C++ compiler for your platform")
  endif()
  hpx_add_config_define(HPX_HAVE_PARCELPORT_MPI)
  if(MSVC)
    # FIXME: add OpenMPI specific flag here for now as the hpx_add_compile_flag()
    # below does not add the extra options to the top level directory
    hpx_add_config_define(OMPI_IMPORTS)
  endif()
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
  if(HPX_WITH_PARCELPORT_MPI_ENV)
    string(REPLACE ";" "," hpx_parcelport_mpi_env_ "${HPX_WITH_PARCELPORT_MPI_ENV}")
    hpx_add_config_define(HPX_HAVE_PARCELPORT_MPI_ENV "\"${hpx_parcelport_mpi_env_}\"")
  endif()
  if(HPX_WITH_PARCELPORT_MPI_MULTITHREADED)
    hpx_add_config_define(HPX_HAVE_PARCELPORT_MPI_MULTITHREADED)
  endif()

  macro(add_parcelport_mpi_module)
    hpx_debug("add_parcelport_mpi_module")
    if(MPI_CXX_INCLUDE_PATH)
      include_directories(${MPI_CXX_INCLUDE_PATH})
    endif()

    set(_mpi_libraries)
    if(MPI_CXX_LIBRARIES)
      set(_mpi_libraries ${_mpi_libraries} ${MPI_CXX_LIBRARIES})
    endif()
    if(MPI_LIBRARY)
      set(_mpi_libraries ${_mpi_libraries} ${MPI_LIBRARY})
    endif()
    if(MPI_EXTRA_LIBRARY)
      set(_mpi_libraries ${_mpi_libraries} ${MPI_EXTRA_LIBRARY})
    endif()
    add_parcelport(mpi
      STATIC
      SOURCES
        "${PROJECT_SOURCE_DIR}/plugins/parcelport/mpi/mpi_environment.cpp"
        "${PROJECT_SOURCE_DIR}/plugins/parcelport/mpi/parcelport_mpi.cpp"
      HEADERS
        "${PROJECT_SOURCE_DIR}/hpx/plugins/parcelport/mpi/header.hpp"
        "${PROJECT_SOURCE_DIR}/hpx/plugins/parcelport/mpi/locality.hpp"
        "${PROJECT_SOURCE_DIR}/hpx/plugins/parcelport/mpi/mpi_environment.hpp"
        "${PROJECT_SOURCE_DIR}/hpx/plugins/parcelport/mpi/sender.hpp"
        "${PROJECT_SOURCE_DIR}/hpx/plugins/parcelport/mpi/receiver.hpp"
      DEPENDENCIES
        ${_mpi_libraries}
      FOLDER "Core/Plugins/Parcelport/MPI")
  endmacro()
else()
  macro(add_parcelport_mpi_module)
  endmacro()
endif()
