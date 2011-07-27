# Copyright (c) 2011 Bryce Lelbach 
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying 
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(type distributed)

set(group examples.math)

add_subdirectory(discovery)
add_subdirectory(integrator)

set(examples
    #rational64
    double)

foreach(example ${examples})
  add_hpx_executable(${type}_solver_${example}
    MODULE math
    SOURCES solver_${example}.cpp
    DEPENDENCIES discovery_component
                 ${example}_integrator_component)

  # depend on the executable
  add_hpx_pseudo_dependencies(${group} ${type}_solver_${example}_exe)
endforeach()

