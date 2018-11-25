# Copyright (c) 2007-2016 Hartmut Kaiser
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

include(HPX_AddLibrary)

if(HPX_WITH_COMPRESSION_SNAPPY)
  find_package(Snappy)
  if(NOT SNAPPY_FOUND)
    hpx_error("Snappy could not be found and HPX_WITH_COMPRESSION_SNAPPY=ON, please specify SNAPPY_ROOT to point to the correct location or set HPX_WITH_COMPRESSION_SNAPPY to OFF")
  endif()
endif()

function(add_snappy_module)
  hpx_debug("add_snappy_module" "SNAPPY_FOUND: ${SNAPPY_FOUND}")
  if(HPX_WITH_COMPRESSION_SNAPPY)
    add_hpx_library(compress_snappy
      PLUGIN
      SOURCES
        "${PROJECT_SOURCE_DIR}/plugins/binary_filter/snappy/snappy_serialization_filter.cpp"
      HEADERS
        "${PROJECT_SOURCE_DIR}/hpx/plugins/binary_filter/snappy_serialization_filter.hpp"
        "${PROJECT_SOURCE_DIR}/hpx/plugins/binary_filter/snappy_serialization_filter_registration.hpp"
      FOLDER "Core/Plugins/Compression"
      DEPENDENCIES ${SNAPPY_LIBRARY})

    target_include_directories(compress_snappy SYSTEM PRIVATE ${SNAPPY_INCLUDE_DIR})

    # FIXME: target_link_directories only exists for CMake >= 3.13
    if(MSVC)
      target_link_directories(compress_snappy PRIVATE ${SNAPPY_LIBRARY_DIR})
    endif()

    target_link_libraries(compress_snappy PRIVATE hpx::boost)

    add_hpx_pseudo_dependencies(plugins.binary_filter.snappy compress_snappy)
    add_hpx_pseudo_dependencies(core plugins.binary_filter.snappy)
  endif()
endfunction()

