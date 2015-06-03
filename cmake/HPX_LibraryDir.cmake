# Copyright (c) 2014 Thomas Heller
# Copyright (c) 2011 Bryce Lelbach
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

if(NOT DEFINED HPX_LIBRARY_DIR)
  set(HPX_LIBRARY_DIR CACHE INTERNAL "" FORCE)
endif()

macro(hpx_library_dir)
  set(_dirs)
  set(_keyword "")
  set(_skip FALSE)
  foreach(dir ${ARGN})
    if(dir STREQUAL "debug" OR
       dir STREQUAL "general" OR
       dir STREQUAL "optimized")
      set(_keyword ${dir})
      set(_skip TRUE)
    else()
      set(_skip FALSE)
    endif()

    if(NOT _skip)
      list(FIND HPX_LIBRARIES "${dir}" _found)
      if(_found EQUAL -1)
        set(_dirs ${_dirs} ${_keyword} ${dir})
      endif()
      set(_keyword "")
      set(_skip FALSE)
    endif()
  endforeach()
  set(HPX_LIBRARY_DIR ${HPX_LIBRARY_DIR} ${_dirs})
  list(SORT HPX_LIBRARY_DIR)
  list(REMOVE_DUPLICATES HPX_LIBRARY_DIR)
  set(HPX_LIBRARY_DIR ${HPX_LIBRARY_DIR} CACHE INTERNAL "" FORCE)
endmacro()
