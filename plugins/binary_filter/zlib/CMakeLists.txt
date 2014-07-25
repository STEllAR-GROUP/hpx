# Copyright (c) 2007-2013 Hartmut Kaiser
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

include(HPX_AddLibrary)

if(HPX_WITH_COMPRESSION_ZLIB)
  find_package(ZLIB)
  if(NOT ZLIB_FOUND)
    hpx_error("zlib could not be found and HPX_WITH_COMPRESSION_ZLIB=ON, please point cmake to the correct location or set HPX_WITH_COMPRESSION_BZIP2 to OFF")
  endif()
endif()

macro(add_zlib_module)
  hpx_debug("add_zlib_module" "ZLIB_FOUND: ${ZLIB_FOUND}")
  if(HPX_WITH_COMPRESSION_ZLIB)
    include_directories(${ZLIB_INCLUDE_DIRS})

    add_hpx_library(compress_zlib
      PLUGIN
      SOURCES "${hpx_SOURCE_DIR}/plugins/binary_filter/zlib/zlib_serialization_filter.cpp"
      HEADERS "${hpx_SOURCE_DIR}/hpx/plugins/binary_filter/zlib_serialization_filter.hpp"
      FOLDER "Core/Plugins/Compression"
      DEPENDENCIES ${ZLIB_LIBRARIES})

    add_hpx_pseudo_dependencies(plugins.binary_filter.zlib compress_zlib_lib)
  endif()
endmacro()
