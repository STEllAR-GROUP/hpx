# Copyright (c) 2011 Bryce Lelbach 
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying 
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(type smp)

set(group examples.math)

set(examples
    naive_ackermann_peter
    predicting_ackermann_peter
    naive_binomial_distribution
    naive_jacobsthal
    naive_lucas_jacobsthal)

foreach(example ${examples})
  add_hpx_executable(${type}_${example}
    MODULE math 
    SOURCES ${example}.cpp)

  # depend on the executable
  add_hpx_pseudo_dependencies(${group} ${type}_${example}_exe)
endforeach()

set(examples
    naive_ackermann_peter_bigint)

foreach(example ${examples})
  add_hpx_executable(${type}_${example}
    MODULE math
    SOURCES ${example}.cpp
    DEPENDENCIES bigint_component)

  # depend on the executable
  add_hpx_pseudo_dependencies(${group} ${type}_${example}_exe)
endforeach()

