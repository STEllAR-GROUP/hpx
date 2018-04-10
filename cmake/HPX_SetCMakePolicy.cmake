# Copyright (c) 2018 Hartmut Kaiser
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

macro(hpx_set_cmake_policy policy value)
  if(POLICY ${policy})
    cmake_policy(SET ${policy} ${value})
  endif()
endmacro()
