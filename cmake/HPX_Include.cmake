# Copyright (c) 2011 Bryce Lelbach
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

macro(hpx_include)
  foreach(listfile ${ARGV})
    string(TOUPPER "HPX_${listfile}_LOADED" detector)
    if(NOT ${detector})
      include("HPX_${listfile}")
    endif()
  endforeach()
endmacro()

