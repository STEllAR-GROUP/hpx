# Copyright (c) 2012 Thomas Heller
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

if(HPX_CXX11_LAMBDAS)

add_definitions(-DHPX_ACTION_ARGUMENT_LIMIT=5)
add_definitions(-DHPX_COMPONENT_CREATE_ARGUMENT_LIMIT=5)
add_definitions(-DHPX_FUNCTION_LIMIT=8)

set(jacobi_sources
    jacobi.cpp
)

set(jacobi_component_sources
    server/solver.cpp
    server/stencil_iterator.cpp
    server/row.cpp
    grid.cpp
    stencil_iterator.cpp
    jacobi_component.cpp
    row.cpp
   )
  
set(jacobi_FLAGS
    DEPENDENCIES iostreams_component
                 jacobi_component
                 distributing_factory_component
)

source_group("Source Files"
             FILES ${jacobi_sources} ${jacobi_component_sources})

add_hpx_component(jacobi
  MODULE jacobi
  SOURCES ${jacobi_component_sources}
  ${jacobi_component_FLAGS}
  FOLDER "Examples/Jacobi")

# add jacobi executable
add_hpx_executable(jacobi
  MODULE jacobi
  SOURCES ${jacobi_sources}
  ${jacobi_FLAGS}
  FOLDER "Examples/Jacobi")

# add a custom target for this example
add_hpx_pseudo_target(examples.jacobi.jacobi)

# make pseudo-targets depend on master pseudo-target
add_hpx_pseudo_dependencies(examples.jacobi
                            examples.jacobi.jacobi)

# add dependencies to pseudo-target
add_hpx_pseudo_dependencies(examples.jacobi.jacobi
                            jacobi_exe)

endif()
