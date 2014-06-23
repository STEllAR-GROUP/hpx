# Copyright (c) 2012 Bryce Adelstein-Lelbach
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(HPX_ISTARGET_LOADED TRUE)

macro(hpx_is_target variable target)

  if(POLICY CMP0026)
    cmake_policy(PUSH)
    cmake_policy(SET CMP0026 OLD)
  endif()

  get_target_property(is_target ${target} LOCATION)

  if(POLICY CMP0026)
    cmake_policy(POP)
  endif()

  if(${is_target} STREQUAL "is_target-NOTFOUND")
    set(${variable} FALSE)
  else()
    set(${variable} TRUE)
  endif()
endmacro()

