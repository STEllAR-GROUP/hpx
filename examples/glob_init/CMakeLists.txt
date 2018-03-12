# Copyright (c) 2018      Nikunj Gupta
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

###############################################################################

add_hpx_executable(global_factorial
  SOURCES factorial.cpp
  FOLDER "Examples/Glob_init/")

add_hpx_pseudo_target(examples.glob_init.global_factorial)
add_hpx_pseudo_dependencies(examples.glob_init examples.glob_init.global_factorial)
add_hpx_pseudo_dependencies(examples.glob_init.global_factorial global_factorial_exe)
