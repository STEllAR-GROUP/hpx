# Copyright (c) 2007-2016 Hartmut Kaiser
# Copyright (c) 2011      Bryce Lelbach
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

if(NOT HPX_WITH_DISTRIBUTED_RUNTIME)
  return()
endif()

set(example_programs accumulator template_accumulator
                     template_function_accumulator
)

# TODO: Add non-interactive version that can be tested.

# for all targets specified above
foreach(example_program ${example_programs})
  set(client_sources ${example_program}_client.cpp)
  set(component_sources ${example_program}.cpp)
  set(component_headers ${example_program}.hpp server/${example_program}.hpp)

  source_group("Source Files" FILES ${client_sources} ${component_sources})

  # add example components
  add_hpx_component(
    ${example_program} INTERNAL_FLAGS NOEXPORT
    SOURCES ${component_sources}
    HEADERS ${component_headers}
    FOLDER "Examples/Quickstart/Accumulators/${example_program}"
  )

  # add example executable
  add_hpx_executable(
    ${example_program} INTERNAL_FLAGS
    SOURCES ${client_sources}
    DEPENDENCIES ${example_program}_component
    FOLDER "Examples/Quickstart/Accumulators/${example_program}"
  )

  add_hpx_example_target_dependencies("accumulators" ${example_program})

endforeach()
