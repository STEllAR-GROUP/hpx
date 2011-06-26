# Copyright (c) 2011 Bryce Lelbach 
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying 
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(type serial)

set(group examples.math)

set(examples
    ackermann_peter
    ackermann_peter_bigint
    binomial_distribution
    factorial
    factorial_bigint
    jacobsthal
    jacobsthal_bigint
    lanczos_gamma
    lucas_jacobsthal
    numerical_integration)

foreach(example ${examples})
  add_hpx_executable(${type}_${example}
    MODULE math
    SOURCES ${example}.cpp
    DEPENDENCIES rt
    NOLIBS)

  # depend on the executable
  add_hpx_pseudo_dependencies(${group} ${type}_${example}_exe)
endforeach()

