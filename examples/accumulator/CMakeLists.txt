# Copyright (c) 2007-2009 Hartmut Kaiser
# Copyright (c) 2011      Bryce Lelbach
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(examples
    accumulator
    template_function_accumulator
   )

# for all targets specified above
foreach(example ${examples})
  set(client_sources ${example}_client.cpp)
  set(component_sources ${example}.cpp)

  source_group("Source Files" FILES ${client_sources} ${component_sources})

  # add example components
  add_hpx_component(${example}
    SOURCES ${component_sources}
    FOLDER "Examples/Quickstart/Accumulators/${example}")

  # add example executable
  add_hpx_executable(${example}_client
    SOURCES ${client_sources}
    DEPENDENCIES ${example}_component
    FOLDER "Examples/Quickstart/Accumulators/${example}")

  # add a custom target for this example
  add_hpx_pseudo_target(examples.accumulator.${example})

  # make pseudo-targets depend on master pseudo-target
  add_hpx_pseudo_dependencies(examples.accumulator
                              examples.accumulator.${example})

  # add dependencies to pseudo-target
  add_hpx_pseudo_dependencies(examples.accumulator.${example}
                              ${example}_client_exe)
endforeach()

