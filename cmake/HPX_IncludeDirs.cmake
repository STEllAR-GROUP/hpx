# Copyright (c) 2014 Thomas Heller
# Copyright (c) 2011 Bryce Lelbach
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

if(NOT DEFINED HPX_INCLUDE_DIRS)
  set(HPX_INCLUDE_DIRS CACHE INTERNAL "" FORCE)
endif()

function(hpx_include_dirs)
  set(_include_dirs)
  set(_keyword "")
  set(_skip FALSE)
  foreach(lib ${ARGN})
    if(lib STREQUAL "debug" OR
       lib STREQUAL "general" OR
       lib STREQUAL "optimized")
      set(_keyword ${lib})
      set(_skip TRUE)
    else()
      set(_skip FALSE)
    endif()

    if(NOT _skip)
      list(FIND HPX_INCLUDE_DIRS "${lib}" _found)
      if(_found EQUAL -1)
        set(_include_dirs ${_include_dirs} ${_keyword} ${lib})
      endif()
      set(_keyword "")
      set(_skip FALSE)
    endif()
  endforeach()
  set(HPX_INCLUDE_DIRS ${HPX_INCLUDE_DIRS} ${_include_dirs} CACHE INTERNAL "" FORCE)
endfunction(hpx_include_dirs)
