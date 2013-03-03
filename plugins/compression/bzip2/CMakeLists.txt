# Copyright (c) 2007-2013 Hartmut Kaiser
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

include(HPX_AddLibrary)

hpx_option(HPX_USE_COMPRESSION_BZIP2 BOOL "Enable bzip2 compression for parcel data (default: OFF)." OFF ADVANCED)
if(HPX_USE_COMPRESSION_BZIP2)
  if(MSVC)
    set(BOOST_LIBRARIES ${BOOST_LIBRARIES} iostreams bzip2)
    set(BZIP2_FOUND ON CACHE INTERNAL "BZIP2_FOUND")
  else()
    find_package(HPX_BZip2)
    if(BZIP2_FOUND)
      set(BOOST_LIBRARIES ${BOOST_LIBRARIES} iostreams)
    endif()
  endif()
endif()

macro(add_bzip2_module)
  hpx_debug("add_bzip2_module" "BZIP2_FOUND: ${BZIP2_FOUND}")
  if(HPX_USE_COMPRESSION_BZIP2 AND BZIP2_FOUND)
    hpx_include_sys_directories(${BZIP2_INCLUDE_DIR})
    hpx_link_sys_directories(${BZIP2_LIBRARY_DIR})

    add_hpx_library(compress_bzip2
      SOURCES ${hpx_SOURCE_DIR}/plugins/compression/bzip2/bzip2_serialization_filter.cpp
      HEADERS ${hpx_SOURCE_DIR}/hpx/plugins/compression/bzip2_serialization_filter.hpp
      FOLDER "Core/Plugins/Compression"
      DEPENDENCIES ${BZIP2_LIBRARY})

    set_property(TARGET compress_bzip2_lib APPEND
      PROPERTY COMPILE_DEFINITIONS
      "HPX_LIBRARY_EXPORTS")

    add_hpx_pseudo_dependencies(plugins.compression.bzip2 compress_bzip2_lib)

    if(NOT HPX_NO_INSTALL)
      hpx_library_install(compress_bzip2_lib lib/hpx)
    endif()
  endif()
endmacro()

