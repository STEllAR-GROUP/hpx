# Copyright (c) 2012 Thomas Heller
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

add_subdirectory(jacobi_component)

set(jacobi_sources
    jacobi.cpp)

set(jacobi_FLAGS
    COMPONENT_DEPENDENCIES
    iostreams jacobi_component)

# add jacobi executable
add_hpx_executable(jacobi
MODULE jacobi
SOURCES ${jacobi_sources}
${jacobi_FLAGS}
FOLDER "Examples/Jacobi")

# add a custom target for this example
add_hpx_pseudo_target(examples.jacobi.jacobi_simple)

# make pseudo-targets depend on master pseudo-target
add_hpx_pseudo_dependencies(examples.jacobi
                            examples.jacobi.jacobi_simple)

# add dependencies to pseudo-target
add_hpx_pseudo_dependencies(examples.jacobi.jacobi_simple
                            jacobi_exe)
