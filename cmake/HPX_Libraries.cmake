# Copyright (c) 2014 Thomas Heller
# Copyright (c) 2011 Bryce Lelbach
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

if(NOT DEFINED HPX_LIBRARIES)
  set(HPX_LIBRARIES CACHE INTERNAL "" FORCE)
endif()

macro(hpx_libraries)
  set(_libs)
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
      list(FIND HPX_LIBRARIES "${lib}" _found)
      if(_found EQUAL -1)
        set(_libs ${_libs} ${_keyword} ${lib})
      endif()
      set(_keyword "")
      set(_skip FALSE)
    endif()
  endforeach()
  set(HPX_LIBRARIES ${HPX_LIBRARIES} ${_libs} CACHE INTERNAL "" FORCE)
endmacro()
