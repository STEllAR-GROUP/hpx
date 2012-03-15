# Copyright (c) 2011 Bryce Lelbach
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

# FIXME: Use parse arg.

set(HPX_OPTION_LOADED TRUE)

macro(hpx_option option type description default)
  if(DEFINED ${option})
    set(${option} "${${option}}" CACHE ${type} ${description} FORCE)
  else()
    set(${option} "${default}" CACHE ${type} ${description} FORCE)
  endif()

  foreach(arg ${ARGN})
    if(arg STREQUAL "ADVANCED")
      mark_as_advanced(FORCE ${option})
    else()
      hpx_error("option" "Unknown argument while calling hpx_option: ${arg} (only allowed value: 'ADVANCED')")
    endif()
  endforeach()
endmacro()

