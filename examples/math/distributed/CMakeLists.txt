# Copyright (c) 2011 Bryce Lelbach 
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying 
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(distributed integration)

set(group examples.math)

set(examples
    trapezoidal_rule)

foreach(example ${examples})
  add_hpx_executable(${type}_${example}
    MODULE math
    SOURCES ${example}.cpp
    DEPENDENCIES csv_parser_lib)

  # depend on the executable
  add_hpx_pseudo_dependencies(${group} ${type}_${example}_exe)
endforeach()

