# Copyright (c) 2007-2013 Hartmut Kaiser
# Copyright (c) 2014-2015 Thomas Heller
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

include(HPX_AddLibrary)

################################################################################
# Decide whether to use the MPI based parcelport
################################################################################
macro(add_parcelport_mpi_module)
  if(HPX_WITH_PARCELPORT_MPI)
    hpx_debug("add_parcelport_mpi_module")
    add_parcelport(mpi
      STATIC
      SOURCES
        "${PROJECT_SOURCE_DIR}/plugins/parcelport/mpi/mpi_environment.cpp"
        "${PROJECT_SOURCE_DIR}/plugins/parcelport/mpi/parcelport_mpi.cpp"
      HEADERS
        "${PROJECT_SOURCE_DIR}/hpx/plugins/parcelport/mpi/header.hpp"
        "${PROJECT_SOURCE_DIR}/hpx/plugins/parcelport/mpi/locality.hpp"
        "${PROJECT_SOURCE_DIR}/hpx/plugins/parcelport/mpi/mpi_environment.hpp"
        "${PROJECT_SOURCE_DIR}/hpx/plugins/parcelport/mpi/receiver.hpp"
        "${PROJECT_SOURCE_DIR}/hpx/plugins/parcelport/mpi/receiver_connection.hpp"
        "${PROJECT_SOURCE_DIR}/hpx/plugins/parcelport/mpi/sender.hpp"
        "${PROJECT_SOURCE_DIR}/hpx/plugins/parcelport/mpi/sender_connection.hpp"
        "${PROJECT_SOURCE_DIR}/hpx/plugins/parcelport/mpi/tag_provider.hpp"
      DEPENDENCIES
        hpx::mpi
      FOLDER "Core/Plugins/Parcelport/MPI")
  endif()
endmacro()
