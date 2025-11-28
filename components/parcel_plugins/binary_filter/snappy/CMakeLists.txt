# Copyright (c) 2007-2022 Hartmut Kaiser
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

if(NOT HPX_WITH_COMPRESSION_SNAPPY)
  return()
endif()

include(HPX_AddLibrary)

find_package(Snappy)
if(NOT Snappy_FOUND)
  hpx_error("Snappy could not be found and HPX_WITH_COMPRESSION_SNAPPY=ON, \
    please specify SNAPPY_ROOT to point to the correct location or set \
    HPX_WITH_COMPRESSION_SNAPPY to OFF"
  )
endif()

hpx_debug("add_snappy_module" "SNAPPY_FOUND: ${Snappy_FOUND}")

add_hpx_library(
  compression_snappy INTERNAL_FLAGS PLUGIN
  SOURCE_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/src"
  SOURCES "snappy_serialization_filter.cpp"
  PREPEND_SOURCE_ROOT
  HEADER_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/include"
  HEADERS "hpx/include/compression_snappy.hpp"
          "hpx/binary_filter/snappy_serialization_filter.hpp"
          "hpx/binary_filter/snappy_serialization_filter_registration.hpp"
  PREPEND_HEADER_ROOT INSTALL_HEADERS ${HPX_WITH_UNITY_BUILD_OPTION}
  ${HPX_WITH_CXX_MODULES_OPTION}
  FOLDER "Core/Plugins/Compression"
  DEPENDENCIES ${Snappy_LIBRARY}
)

target_include_directories(
  compression_snappy SYSTEM PRIVATE ${Snappy_INCLUDE_DIR}
)
target_link_directories(compression_snappy PRIVATE ${Snappy_LIBRARY_DIR})

target_link_libraries(
  compression_snappy PUBLIC Boost::iostreams ${SNAPPY_LIBRARY}
)

add_hpx_pseudo_dependencies(
  components.parcel_plugins.binary_filter.snappy compression_snappy
)
add_hpx_pseudo_dependencies(
  components components.parcel_plugins.binary_filter.snappy
)

add_subdirectory(tests)
