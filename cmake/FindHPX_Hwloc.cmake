
# Copyright (c) 2012 Thomas Heller
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

if(NOT HPX_FINDPACKAGE_LOADED)
  include(HPX_FindPackage)
endif(NOT HPX_FINDPACKAGE_LOADED)

hpx_find_package(HWLOC
  LIBRARIES hwloc
  LIBRARY_PATHS lib64 lib
  HEADERS hwloc.h
  HEADER_PATHS include
)

if(MSVC AND NOT HWLOC_FOUND)
  # the binary distribution of hwloc has strange naming conventions for the
  # library file
  hpx_find_package(HWLOC
    LIBRARIES libhwloc
    LIBRARY_PATHS lib64 lib
    HEADERS hwloc.h
    HEADER_PATHS include
  )
endif(MSVC AND NOT HWLOC_FOUND)
