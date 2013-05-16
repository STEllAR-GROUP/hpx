# Copyright (c) 2007-2013 Hartmut Kaiser
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

include(HPX_AddLibrary)

hpx_option(HPX_USE_COMPRESSION_ZLIB BOOL "Enable zlib compression for parcel data (default: OFF)." OFF ADVANCED)
if(HPX_USE_COMPRESSION_ZLIB)
  if(MSVC)
    set(BOOST_LIBRARIES ${BOOST_LIBRARIES} iostreams zlib)
    set(ZLIB_FOUND ON CACHE INTERNAL "ZLIB_FOUND")
  else()
    find_package(HPX_Zlib)
    if(ZLIB_FOUND)
      set(BOOST_LIBRARIES ${BOOST_LIBRARIES} iostreams)
    endif()
  endif()
endif()

macro(add_zlib_module)
  hpx_debug("add_zlib_module" "ZLIB_FOUND: ${ZLIB_FOUND}")
  if(HPX_USE_COMPRESSION_ZLIB AND ZLIB_FOUND)
    hpx_include_sys_directories(${ZLIB_INCLUDE_DIR})
    hpx_link_sys_directories(${ZLIB_LIBRARY_DIR})

    add_hpx_library(compress_zlib
      SOURCES ${hpx_SOURCE_DIR}/plugins/compression/zlib/zlib_serialization_filter.cpp
      HEADERS ${hpx_SOURCE_DIR}/hpx/plugins/compression/zlib_serialization_filter.hpp
      FOLDER "Core/Plugins/Compression"
      DEPENDENCIES ${ZLIB_LIBRARY})

    set_property(TARGET compress_zlib_lib APPEND
      PROPERTY COMPILE_DEFINITIONS
      "HPX_LIBRARY_EXPORTS"
      "HPX_PLUGIN_NAME=compress_zlib")

    add_hpx_pseudo_dependencies(plugins.compression.zlib compress_zlib_lib)

    if(NOT HPX_NO_INSTALL)
      hpx_library_install(compress_zlib_lib lib/hpx)
    endif()
  endif()
endmacro()
