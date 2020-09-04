# Copyright (c) 2007-2016 Hartmut Kaiser
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

include(HPX_AddLibrary)

if(HPX_WITH_COMPRESSION_BZIP2)
  if(MSVC)
    set(BOOST_LIBRARIES ${BOOST_LIBRARIES} iostreams bzip2)
    set(BZIP2_FOUND
        ON
        CACHE INTERNAL "BZIP2_FOUND"
    )
  else()
    find_package(BZip2)
    if(NOT BZIP2_FOUND)
      hpx_error(
        "BZip2 could not be found and HPX_WITH_COMPRESSION_BZIP2=ON, please specify BZIP2_ROOT to point to the correct location or set HPX_WITH_COMPRESSION_BZIP2 to OFF"
      )
    endif()
  endif()

  hpx_debug("add_bzip2_module" "BZIP2_FOUND: ${BZIP2_FOUND}")

  set(SOURCE_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/src")
  set(HEADER_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/include")

  add_hpx_library(
    compression_bzip2 INTERNAL_FLAGS PLUGIN
    SOURCES "${SOURCE_ROOT}/bzip2_serialization_filter.cpp"
    HEADERS
      "${HEADER_ROOT}/hpx/include/compression_bzip2.hpp"
      "${HEADER_ROOT}/hpx/plugins/binary_filter/bzip2_serialization_filter.hpp"
      "${HEADER_ROOT}/hpx/plugins/binary_filter/bzip2_serialization_filter_registration.hpp"
    FOLDER "Core/Plugins/Compression"
    DEPENDENCIES ${BZIP2_LIBRARIES} ${HPX_WITH_UNITY_BUILD_OPTION}
  )

  if(NOT MSVC)
    target_include_directories(
      compression_bzip2 SYSTEM PRIVATE ${BZIP2_INCLUDE_DIR}
    )
  endif()

  target_include_directories(
    compression_bzip2 PUBLIC $<BUILD_INTERFACE:${HEADER_ROOT}>
  )
  target_link_libraries(compression_bzip2 PUBLIC Boost::iostreams)

  add_hpx_pseudo_dependencies(plugins.binary_filter.bzip2 compression_bzip2)
  add_hpx_pseudo_dependencies(core plugins.binary_filter.bzip2)
endif()
