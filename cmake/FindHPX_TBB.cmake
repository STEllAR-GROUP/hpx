# Copyright (c) 2011-2012 Bryce Adelstein-Lelbach
# Copyright (c) 2011-2013 Thomas Heller
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

if(NOT HPX_FINDPACKAGE_LOADED)
  include(HPX_FindPackage)
endif()

set(TBB_LIB_SEARCH_PATH "")

if(HPX_NATIVE_MIC)
  set(TBB_LIB_SEARCH_PATH "lib/mic")
else()
  set(TBB_LIB_SEARCH_PATH "lib/intel64")
endif()

hpx_find_package(TBB
  LIBRARIES tbb libtbb
  LIBRARY_PATHS lib64 lib ${TBB_LIB_SEARCH_PATH}
  HEADERS tbb/tbb.h
  HEADER_PATHS include)

