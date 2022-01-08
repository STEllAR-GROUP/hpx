# Copyright (c) 2007-2021 Hartmut Kaiser
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

include(HPX_AddLibrary)

if(HPX_WITH_COMPRESSION_SNAPPY)
  find_package(Snappy)
  if(NOT SNAPPY_FOUND)
    hpx_error("Snappy could not be found and HPX_WITH_COMPRESSION_SNAPPY=ON, \
      please specify SNAPPY_ROOT to point to the correct location or set \
      HPX_WITH_COMPRESSION_SNAPPY to OFF"
    )
  endif()

  hpx_debug("add_snappy_module" "SNAPPY_FOUND: ${SNAPPY_FOUND}")

  add_hpx_library(
    compression_snappy INTERNAL_FLAGS PLUGIN
    SOURCE_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/src"
    SOURCES "snappy_serialization_filter.cpp"
    PREPEND_SOURCE_ROOT
    HEADER_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/include"
    HEADERS
      "hpx/include/compression_snappy.hpp"
      "hpx/plugins/binary_filter/snappy_serialization_filter.hpp"
      "hpx/plugins/binary_filter/snappy_serialization_filter_registration.hpp"
    PREPEND_HEADER_ROOT INSTALL_HEADERS
    FOLDER "Core/Plugins/Compression"
    DEPENDENCIES ${SNAPPY_LIBRARY} ${HPX_WITH_UNITY_BUILD_OPTION}
  )

  target_include_directories(
    compression_snappy SYSTEM PRIVATE ${SNAPPY_INCLUDE_DIR}
  )
  target_include_directories(
    compression_snappy PUBLIC $<BUILD_INTERFACE:${HEADER_ROOT}>
  )

  if(MSVC)
    target_link_directories(compression_snappy PRIVATE ${SNAPPY_LIBRARY_DIR})
  endif()

  target_link_libraries(compression_snappy PUBLIC Boost::iostreams)

  add_hpx_pseudo_dependencies(plugins.binary_filter.snappy compression_snappy)
  add_hpx_pseudo_dependencies(core plugins.binary_filter.snappy)
endif()
