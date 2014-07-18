# Copyright (c) 2014 Thomas Heller
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(HPX_EXPORT_TARGETS "" CACHE INTERNAL "" FORCE)

macro(hpx_export_targets)
  foreach(target ${ARGN})
    list(FIND HPX_EXPORT_TARGETS ${target} _found)
    if(_found EQUAL -1)
      set(HPX_EXPORT_TARGETS ${HPX_EXPORT_TARGETS} ${target} CACHE INTERNAL "" FORCE)
    endif()
  endforeach()
endmacro()
