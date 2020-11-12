# Copyright (c) 2014-2016 John Biddiscombe
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

# Configuring options
include(HPX_SetupLibfabric)

if(HPX_WITH_PARCELPORT_LIBFABRIC)
  # ------------------------------------------------------------------------------
  # Define the ParcelPort registration macro
  # ------------------------------------------------------------------------------
  # make these variables cache ones so that the macro definition picks them up
  # correctly it is executed from a different source dir, so vars lose their
  # scope

  set(_parcelport_CXX_srcs
      "${PROJECT_SOURCE_DIR}/plugins/parcelport/libfabric/parcelport_libfabric.cpp"
      "${PROJECT_SOURCE_DIR}/plugins/parcelport/libfabric/receiver.cpp"
      "${PROJECT_SOURCE_DIR}/plugins/parcelport/libfabric/rma_receiver.cpp"
      "${PROJECT_SOURCE_DIR}/plugins/parcelport/libfabric/sender.cpp"
      CACHE INTERNAL "" FORCE
  )

  set(_hpxc_CXX_srcs
      ""
      CACHE INTERNAL "" FORCE
  )

  set(_libfabric_libraries Libfabric::libfabric Pmi::pmi)

  hpx_debug("add_parcelport_libfabric_module")

  include(HPX_AddParcelport)
  add_parcelport(
    libfabric STATIC
    SOURCES ${_parcelport_CXX_srcs}
    # do we need to declare any headers? HEADERS FIXME : Could this be private
    DEPENDENCIES
      ${_libfabric_libraries}
      hpx_performance_counters
      hpx_program_options
      hpx_runtime_local
      hpx_threadmanager
      hpx_parallelism
      hpx_core
    INCLUDE_DIRS ${_libfabric_include_dirs} ${PROJECT_SOURCE_DIR}
                 ${PROJECT_BINARY_DIR}
    FOLDER "Core/Plugins/Parcelport/libfabric"
  )

  set_source_files_properties(
    ${_parcelport_CXX_srcs}
    PROPERTIES COMPILE_FLAGS "-Wno-missing-field-initializers" LANGUAGE CXX
  )

endif()
