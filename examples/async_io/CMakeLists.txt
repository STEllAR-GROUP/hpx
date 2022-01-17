# Copyright (c) 2011 Bryce Lelbach
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

if(NOT HPX_WITH_DISTRIBUTED_RUNTIME)
  return()
endif()

set(example_programs async_io_simple async_io_action async_io_external
                     async_io_low_level
)

set(async_io_simple_PARAMETERS THREADS_PER_LOCALITY 4)
set(async_io_action_PARAMETERS THREADS_PER_LOCALITY 4)
set(async_io_external_PARAMETERS THREADS_PER_LOCALITY 4)
set(async_io_low_level_PARAMETERS THREADS_PER_LOCALITY 4)

foreach(example_program ${example_programs})

  set(${example_program}_FLAGS DEPENDENCIES iostreams_component)

  set(sources ${example_program}.cpp)

  source_group("Source Files" FILES ${sources})

  # add example executable
  add_hpx_executable(
    ${example_program} INTERNAL_FLAGS
    SOURCES ${sources} ${${example_program}_FLAGS}
    FOLDER "Examples/AsyncIO/${example_program}"
  )

  add_hpx_example_target_dependencies("async_io" ${example_program})

  if(HPX_WITH_TESTS AND HPX_WITH_TESTS_EXAMPLES)
    add_hpx_example_test(
      "async_io" ${example_program} ${${example_program}_PARAMETERS}
    )
  endif()
endforeach()
