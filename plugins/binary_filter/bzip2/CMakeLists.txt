# Copyright (c) 2007-2013 Hartmut Kaiser
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

include(HPX_AddLibrary)

if(HPX_WITH_COMPRESSION_BZIP2)
  find_package(BZip2)
  if(NOT BZIP2_FOUND)
    hpx_error("BZip2 could not be found and HPX_WITH_COMPRESSION_BZIP2=ON, please specify BZIP2_ROOT to point to the correct location or set HPX_WITH_COMPRESSION_BZIP2 to OFF")
  endif()
endif()

macro(add_bzip2_module)
  hpx_debug("add_bzip2_module" "BZIP2_FOUND: ${BZIP2_FOUND}")
  if(HPX_WITH_COMPRESSION_BZIP2)
    include_directories(${BZIP2_INCLUDE_DIR})

    add_hpx_library(compress_bzip2
      PLUGIN
      SOURCES "${hpx_SOURCE_DIR}/plugins/binary_filter/bzip2/bzip2_serialization_filter.cpp"
      HEADERS "${hpx_SOURCE_DIR}/hpx/plugins/binary_filter/bzip2_serialization_filter.hpp"
      FOLDER "Core/Plugins/Compression"
      DEPENDENCIES ${BZIP2_LIBRARIES})

    add_hpx_pseudo_dependencies(plugins.binary_filter.bzip2 compress_bzip2_lib)
  endif()
endmacro()
