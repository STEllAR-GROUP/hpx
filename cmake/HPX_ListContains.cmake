# Copyright (c) 2007-2012 Hartmut Kaiser
# Copyright (c) 2011      Bryce Lelbach
# Copyright (c) 2010-2011 Alexander Neundorf
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(HPX_LISTCONTAINS_LOADED TRUE)

macro(hpx_list_contains output needle)
  set(${output})
  foreach(value ${ARGN})
    if(${needle} STREQUAL ${value})
      set(${output} TRUE)
    endif()
  endforeach()
endmacro()

