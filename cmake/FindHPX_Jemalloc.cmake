# Copyright (c) 2011 Bryce Lelbach
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

if(NOT HPX_FINDPACKAGE_LOADED)
  include(HPX_FindPackage)
endif()

hpx_find_package(JEMALLOC
  LIBRARIES jemalloc libjemalloc
  LIBRARY_PATHS lib64 lib
  HEADERS jemalloc/jemalloc.h
  HEADER_PATHS include)

