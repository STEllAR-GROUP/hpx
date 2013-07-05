# Copyright (c) 2011-2012 Bryce Adelstein-Lelbach
# Copyright (c) 2011-2013 Thomas Heller
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

if(NOT HPX_FINDPACKAGE_LOADED)
  include(HPX_FindPackage)
endif()

if(TBB_ROOT AND NOT TBBMALLOC_ROOT)
  set(TBBMALLOC_ROOT ${TBB_ROOT})
endif()

if(TBBMALLOC_ROOT AND NOT TBBMALLOC_PROXY_ROOT)
  set(TBBMALLOC_PROXY_ROOT ${TBBMALLOC_ROOT})
endif()

set(TBBMALLOC_LIB_SEARCH_PATH "")

if(HPX_NATIVE_MIC)
  set(TBBMALLOC_LIB_SEARCH_PATH "lib/mic")
else()
  set(TBBMALLOC_LIB_SEARCH_PATH "lib/intel64")
endif()

hpx_find_package(TBBMALLOC_PROXY
  LIBRARIES tbbmalloc_proxy libtbbmalloc_proxy
  LIBRARY_PATHS lib64 lib ${TBBMALLOC_LIB_SEARCH_PATH}
  HEADERS tbb/scalable_allocator.h
  HEADER_PATHS include)

hpx_find_package(TBBMALLOC
  LIBRARIES tbbmalloc libtbbmalloc
  LIBRARY_PATHS lib64 lib ${TBBMALLOC_LIB_SEARCH_PATH}
  HEADERS tbb/scalable_allocator.h
  HEADER_PATHS include)

if(NOT TBBMALLOC_PROXY_FOUND)
  set(TBBMALLOC_FOUND Off)
endif()
