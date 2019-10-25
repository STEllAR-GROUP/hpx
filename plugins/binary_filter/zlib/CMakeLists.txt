# Copyright (c) 2007-2016 Hartmut Kaiser
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

if(HPX_WITH_COMPRESSION_ZLIB)

  if(MSVC)
    set(BOOST_LIBRARIES ${BOOST_LIBRARIES} iostreams zlib)
    set(ZLIB_FOUND ON CACHE INTERNAL "ZLIB_FOUND")

  else()
    find_package(ZLIB)
    if(NOT ZLIB_FOUND)
      hpx_error("zlib could not be found and HPX_WITH_COMPRESSION_ZLIB=ON, \
        please point cmake to the correct location or set \
        HPX_WITH_COMPRESSION_BZIP2 to OFF")
    endif()

  endif()

  hpx_debug("add_zlib_module" "ZLIB_FOUND: ${ZLIB_FOUND}")

  include(HPX_AddLibrary)
  add_hpx_library(compress_zlib
    PLUGIN
    SOURCES
      "${PROJECT_SOURCE_DIR}/plugins/binary_filter/zlib/zlib_serialization_filter.cpp"
    HEADERS
      "${PROJECT_SOURCE_DIR}/hpx/plugins/binary_filter/zlib_serialization_filter.hpp"
      "${PROJECT_SOURCE_DIR}/hpx/plugins/binary_filter/zlib_serialization_filter_registration.hpp"
    FOLDER "Core/Plugins/Compression"
    DEPENDENCIES ${ZLIB_LIBRARIES})

  if(NOT MSVC)
    target_include_directories(compress_zlib SYSTEM PRIVATE ${ZLIB_INCLUDE_DIRS})
  endif()

  target_link_libraries(compress_zlib PRIVATE hpx::boost::iostreams)

  add_hpx_pseudo_dependencies(plugins.binary_filter.zlib compress_zlib)
  add_hpx_pseudo_dependencies(core plugins.binary_filter.zlib)

endif()
