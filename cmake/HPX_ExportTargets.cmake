# Copyright (c) 2014 Thomas Heller
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

function(hpx_export_targets)
  foreach(target ${ARGN})
    list(FIND HPX_EXPORT_TARGETS ${target} _found)
    if(_found EQUAL -1)
      set(HPX_EXPORT_TARGETS
          ${HPX_EXPORT_TARGETS} ${target}
          CACHE INTERNAL "" FORCE
      )
    endif()
  endforeach()
endfunction(hpx_export_targets)

function(hpx_export_internal_targets)
  foreach(target ${ARGN})
    list(FIND HPX_EXPORT_INTERNAL_TARGETS ${target} _found)
    if(_found EQUAL -1)
      set(HPX_EXPORT_INTERNAL_TARGETS
          ${HPX_EXPORT_INTERNAL_TARGETS} ${target}
          CACHE INTERNAL "" FORCE
      )
    endif()
  endforeach()
endfunction(hpx_export_internal_targets)
