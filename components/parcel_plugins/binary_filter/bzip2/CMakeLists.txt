# Copyright (c) 2007-2022 Hartmut Kaiser
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

if(NOT HPX_WITH_COMPRESSION_BZIP2)
  return()
endif()

include(HPX_AddLibrary)

find_package(BZip2)
if(NOT BZIP2_FOUND)
  hpx_error("BZip2 could not be found and HPX_WITH_COMPRESSION_BZIP2=ON, \
    please specify BZIP2_ROOT to point to the correct location or set \
    HPX_WITH_COMPRESSION_BZIP2 to OFF"
  )
endif()

hpx_debug("add_bzip2_module" "BZIP2_FOUND: ${BZIP2_FOUND}")

add_hpx_library(
  compression_bzip2 INTERNAL_FLAGS PLUGIN
  SOURCE_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/src"
  SOURCES "bzip2_serialization_filter.cpp"
  PREPEND_SOURCE_ROOT
  HEADER_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/include"
  HEADERS "hpx/include/compression_bzip2.hpp"
          "hpx/binary_filter/bzip2_serialization_filter.hpp"
          "hpx/binary_filter/bzip2_serialization_filter_registration.hpp"
  PREPEND_HEADER_ROOT INSTALL_HEADERS ${HPX_WITH_UNITY_BUILD_OPTION}
  ${HPX_WITH_CXX_MODULES_OPTION}
  FOLDER "Core/Plugins/Compression"
)

target_include_directories(
  compression_bzip2 SYSTEM PRIVATE ${BZIP2_INCLUDE_DIR}
)
target_link_libraries(
  compression_bzip2 PUBLIC Boost::iostreams ${BZIP2_LIBRARIES}
)

add_hpx_pseudo_dependencies(
  components.parcel_plugins.binary_filter.bzip2 compression_bzip2
)
add_hpx_pseudo_dependencies(components.parcel_plugins.binary_filter.bzip2)

add_subdirectory(tests)
