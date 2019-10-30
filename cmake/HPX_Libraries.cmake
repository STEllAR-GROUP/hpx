# Copyright (c) 2014 Thomas Heller
# Copyright (c) 2011 Bryce Lelbach
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

if(NOT DEFINED HPX_BASE_LIBRARIES)
  set(HPX_BASE_LIBRARIES CACHE INTERNAL "" FORCE)
endif()

# This function aims at linking the base libraries to the hpx target
function(hpx_base_libraries)
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
      list(FIND HPX_BASE_LIBRARIES "${lib}" _found)
      if(_found EQUAL -1)
        set(_libs ${_libs} ${_keyword} ${lib})
      endif()
      set(_keyword "")
      set(_skip FALSE)
    endif()
  endforeach()
  set(HPX_BASE_LIBRARIES ${HPX_BASE_LIBRARIES} ${_libs} CACHE INTERNAL "" FORCE)
endfunction(hpx_base_libraries)

# FIXME : hpx_libraries is temporary, it's to deprecate progressively the
# libraries variables (use of target_link_libraries directives instead)
if(NOT DEFINED HPX_LIBRARIES)
  set(HPX_LIBRARIES CACHE INTERNAL "" FORCE)
endif()

function(hpx_libraries)
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
endfunction(hpx_libraries)
