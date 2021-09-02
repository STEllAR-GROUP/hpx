# Copyright (c) 2011 Bryce Lelbach
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(HPXLocal_ADDPSEUDODEPENDENCIES_LOADED TRUE)

include(HPXLocal_Message)

function(hpx_local_add_pseudo_dependencies)

  if("${HPXLocal_CMAKE_LOGLEVEL}" MATCHES "DEBUG|debug|Debug")
    set(args)
    foreach(arg ${ARGV})
      set(args "${args} ${arg}")
    endforeach()
    hpx_local_debug("hpx_local_add_pseudo_dependencies" ${args})
  endif()

  if(HPXLocal_WITH_PSEUDO_DEPENDENCIES)
    set(shortened_args)
    foreach(arg ${ARGV})
      hpx_local_shorten_pseudo_target(${arg} shortened_arg)
      set(shortened_args ${shortened_args} ${shortened_arg})
    endforeach()
    add_dependencies(${shortened_args})
  endif()
endfunction()

function(hpx_local_add_pseudo_dependencies_no_shortening)

  if("${HPXLocal_CMAKE_LOGLEVEL}" MATCHES "DEBUG|debug|Debug")
    set(args)
    foreach(arg ${ARGV})
      set(args "${args} ${arg}")
    endforeach()
    hpx_local_debug("hpx_local_add_pseudo_dependencies" ${args})
  endif()

  if(HPXLocal_WITH_PSEUDO_DEPENDENCIES)
    add_dependencies(${ARGV})
  endif()
endfunction()
