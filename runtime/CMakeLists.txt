# Copyright (c) 2007-2009 Hartmut Kaiser
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying 
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

add_hpx_executable(hpx_runtime
  MODULE core
  SOURCES hpx_runtime.cpp)

add_hpx_pseudo_target(runtime)

add_hpx_pseudo_dependencies(runtime hpx_runtime_exe)

