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
  if(BZIP2_FOUND)
    hpx_add_definitions(-DHPX_LIBRARY_EXPORTS)

    hpx_include_sys_directories(${BZIP2_INCLUDE_DIR})
    hpx_link_sys_directories(${BZIP2_LIBRARY_DIR})

    add_hpx_library(plugin_bzip2
      SOURCES ${hpx_SOURCE_DIR}/plugins/compression/bzip2/bzip2_serialization_filter.cpp
      HEADERS ${hpx_SOURCE_DIR}/hpx/plugins/compression/bzip2_serialization_filter.hpp
      FOLDER "Core/Plugins/Compression"
      DEPENDENCIES ${BZIP2_LIBRARY})

    add_hpx_pseudo_dependencies(plugins.compression.bzip2 plugin_bzip2_lib)
  endif()
endmacro()

