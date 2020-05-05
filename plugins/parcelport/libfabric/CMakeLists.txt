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
  set(_libfabric_C_srcs
      ""
      CACHE INTERNAL "" FORCE
  )

  if(NOT HPX_PARCELPORT_LIBFABRIC_WITH_HPXC)
    set(_libfabric_libraries Libfabric::libfabric Pmi::pmi)
  else()
    add_library(Gni::gni INTERFACE IMPORTED)
    target_include_directories(Gni::gni INTERFACE ${LIBFABRIC_INCLUDE_DIR})
    target_link_libraries(Gni::gni INTERFACE ${GNI_LIB})
    set(_deps Gni::gni Pmi::pmi)

    set(_libfabric_C_srcs ${libfabric_SRCS} ${libfabric_util_SRCS}
                          ${libfabric_gni_SRCS}
    )

    set(_hpxc_CXX_srcs
        ${_parcelport_CXX_srcs}
        "${PROJECT_SOURCE_DIR}/plugins/parcelport/libfabric/hpxc/src/threads/thread.cpp"
    )

    set(_parcelport_CXX_srcs ${_parcelport_CXX_srcs} ${_hpxc_CXX_srcs})

    set(_libfabric_include_dirs
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/hpxc
        ${CMAKE_CURRENT_SOURCE_DIR}/libfabric
        ${CMAKE_CURRENT_SOURCE_DIR}/libfabric/prov/gni
        ${CMAKE_CURRENT_SOURCE_DIR}/libfabric/prov/gni/include
        ${CMAKE_CURRENT_SOURCE_DIR}/libfabric/include
        ${GNI_INCLUDE}
    )
  endif()

  hpx_debug("add_parcelport_libfabric_module")

  include(HPX_AddParcelport)
  add_parcelport(
    libfabric STATIC
    SOURCES ${_parcelport_CXX_srcs} ${_libfabric_C_srcs}
    # do we need to declare any headers? HEADERS FIXME : Could this be private
    DEPENDENCIES
      ${_libfabric_libraries}
      hpx_allocator_support
      hpx_asio
      hpx_assertion
      hpx_cache
      hpx_concurrency
      hpx_config
      hpx_coroutines
      hpx_errors
      hpx_execution
      hpx_hardware
      hpx_memory
      hpx_performance_counters
      hpx_plugin
      hpx_program_options
      hpx_runtime_local
      hpx_serialization
      hpx_threading
      hpx_threadmanager
      hpx_timing
      hpx_topology
    INCLUDE_DIRS ${_libfabric_include_dirs} ${PROJECT_SOURCE_DIR}
                 ${PROJECT_BINARY_DIR}
    FOLDER "Core/Plugins/Parcelport/libfabric"
  )

  if(HPX_PARCELPORT_LIBFABRIC_WITH_HPXC)
    set_source_files_properties(
      ${_libfabric_C_srcs}
      PROPERTIES COMPILE_FLAGS "-include \"hpxc/hpxc/pthread_compatibility.h\" \
            -include \"libfabric/config.h\"                 \
            -Wsign-compare -Wunused-parameter" LANGUAGE C
    )

    set_source_files_properties(
      ${_parcelport_CXX_srcs}
      PROPERTIES
        COMPILE_FLAGS
        "-include \"libfabric/config.h\" -Wno-missing-field-initializers"
        LANGUAGE CXX
    )
  else()
    set_source_files_properties(
      ${_parcelport_CXX_srcs}
      PROPERTIES COMPILE_FLAGS "-Wno-missing-field-initializers" LANGUAGE CXX
    )
  endif()

endif()
