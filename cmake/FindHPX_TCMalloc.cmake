# Copyright (c) 2007-2012 Hartmut Kaiser
# Copyright (c) 2010-2011 Matt Anderson
# Copyright (c) 2011      Bryce Lelbach
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

if(NOT HPX_FINDPACKAGE_LOADED)
  include(HPX_FindPackage)
endif()

if(GOOGLE_PERFTOOLS_FOUND)
  hpx_find_package(TCMALLOC
    LIBRARIES tcmalloc libtcmalloc
    LIBRARY_PATHS lib64 lib)
else()
  hpx_find_package(TCMALLOC
    LIBRARIES tcmalloc_minimal libtcmalloc_minimal
    LIBRARY_PATHS lib64 lib)
endif()

