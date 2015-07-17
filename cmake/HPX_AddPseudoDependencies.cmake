# Copyright (c) 2011 Bryce Lelbach
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(HPX_ADDPSEUDODEPENDENCIES_LOADED TRUE)

include(HPX_Include)

hpx_include(Message)

macro(add_hpx_pseudo_dependencies)

  if("${HPX_CMAKE_LOGLEVEL}" MATCHES "DEBUG|debug|Debug")
    set(args)
    foreach(arg ${ARGV})
      set(args "${args} ${arg}")
    endforeach()
    hpx_debug("add_hpx_pseudo_dependencies" ${args})
  endif()

  if(HPX_WITH_PSEUDO_DEPENDENCIES)
    set(shortened_args)
    foreach(arg ${ARGV})
      shorten_hpx_pseudo_target(${arg} shortened_arg)
      set(shortened_args ${shortened_args} ${shortened_arg})
    endforeach()
    add_dependencies(${shortened_args})
  endif()
endmacro()

macro(add_hpx_pseudo_dependencies_no_shortening)

  if("${HPX_CMAKE_LOGLEVEL}" MATCHES "DEBUG|debug|Debug")
    set(args)
    foreach(arg ${ARGV})
      set(args "${args} ${arg}")
    endforeach()
    hpx_debug("add_hpx_pseudo_dependencies" ${args})
  endif()

  if(HPX_WITH_PSEUDO_DEPENDENCIES)
    add_dependencies(${ARGV})
  endif()
endmacro()

