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

  if(NOT WIN32)
    add_dependencies(${ARGV})
  endif()
endmacro()

