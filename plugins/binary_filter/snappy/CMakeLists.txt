# Copyright (c) 2007-2013 Hartmut Kaiser
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

include(HPX_AddLibrary)

if(HPX_WITH_COMPRESSION_SNAPPY)
  find_package(Snappy)
  if(NOT SNAPPY_FOUND)
    hpx_error("Snappy could not be found and HPX_WITH_COMPRESSION_SNAPPY=ON, please specify SNAPPY_ROOT to point to the correct location or set HPX_WITH_COMPRESSION_SNAPPY to OFF")
  endif()
endif()

macro(add_snappy_module)
  hpx_debug("add_snappy_module" "SNAPPY_FOUND: ${SNAPPY_FOUND}")
  if(HPX_WITH_COMPRESSION_SNAPPY)
    include_directories("${SNAPPY_INCLUDE_DIR}")
    if(MSVC)
      link_directories("${SNAPPY_LIBRARY_DIR}")
    endif()

    add_hpx_library(compress_snappy
      PLUGIN
      SOURCES "${PROJECT_SOURCE_DIR}/plugins/binary_filter/snappy/snappy_serialization_filter.cpp"
      HEADERS "${PROJECT_SOURCE_DIR}/hpx/plugins/binary_filter/snappy_serialization_filter.hpp"
      FOLDER "Core/Plugins/Compression"
      DEPENDENCIES ${SNAPPY_LIBRARY})

    add_hpx_pseudo_dependencies(plugins.binary_filter.snappy compress_snappy_lib)
    add_hpx_pseudo_dependencies(core plugins.binary_filter.snappy)
  endif()
endmacro()

