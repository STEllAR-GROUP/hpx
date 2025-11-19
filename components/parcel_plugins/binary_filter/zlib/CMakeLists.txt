# Copyright (c) 2007-2022 Hartmut Kaiser
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

if(NOT HPX_WITH_COMPRESSION_ZLIB)
  return()
endif()

include(HPX_AddLibrary)

find_package(ZLIB)
if(NOT ZLIB_FOUND)
  hpx_error("zlib could not be found and HPX_WITH_COMPRESSION_ZLIB=ON, \
    please specify ZLIB_ROOT to point to the correct location  \
    HPX_WITH_COMPRESSION_ZLIB to OFF"
  )
endif()

hpx_debug("add_zlib_module" "ZLIB_FOUND: ${ZLIB_FOUND}")

add_hpx_library(
  compression_zlib INTERNAL_FLAGS PLUGIN
  SOURCE_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/src"
  SOURCES "zlib_serialization_filter.cpp"
  PREPEND_SOURCE_ROOT
  HEADER_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/include"
  HEADERS "hpx/include/compression_zlib.hpp"
          "hpx/binary_filter/zlib_serialization_filter.hpp"
          "hpx/binary_filter/zlib_serialization_filter_registration.hpp"
  PREPEND_HEADER_ROOT INSTALL_HEADERS ${HPX_WITH_UNITY_BUILD_OPTION}
  ${HPX_WITH_CXX_MODULES_OPTION}
  FOLDER "Core/Plugins/Compression"
)

target_include_directories(compression_zlib SYSTEM PRIVATE ${ZLIB_INCLUDE_DIRS})
target_link_libraries(
  compression_zlib PUBLIC Boost::iostreams ${ZLIB_LIBRARIES}
)

add_hpx_pseudo_dependencies(
  components.parcel_plugins.binary_filter.zlib compression_zlib
)
add_hpx_pseudo_dependencies(
  components components.parcel_plugins.binary_filter.zlib
)

add_subdirectory(tests)
