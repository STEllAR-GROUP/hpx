# Copyright (c) 2019 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(examples
    config_file_types
    custom_syntax
    env_options
    first
    multiple_sources
    option_groups
    options_description
    options_hierarchy
    real
    regex
    response_file
)

set(disabled_tests)

if(HPX_WITH_EXAMPLES)
  add_hpx_pseudo_target(examples.modules.program_options)
  add_hpx_pseudo_dependencies(examples.modules examples.modules.program_options)

  if(HPX_WITH_TESTS
     AND HPX_WITH_TESTS_EXAMPLES
     AND HPX_PROGRAM_OPTIONS_WITH_TESTS
  )
    add_hpx_pseudo_target(tests.examples.modules.program_options)
    add_hpx_pseudo_dependencies(
      tests.examples.modules tests.examples.modules.program_options
    )
  endif()

  foreach(example ${examples})

    set(sources ${example}.cpp)

    source_group("Source Files" FILES ${sources})

    set(folder_name "Examples/Modules/Full/ProgramOptions")

    # add example executable
    add_hpx_executable(
      ${example} INTERNAL_FLAGS
      SOURCES ${sources} ${${example}_FLAGS}
      DEPENDENCIES ${${example}_LIBS}
      FOLDER ${folder_name}
    )

    add_hpx_example_target_dependencies(
      "modules.program_options" ${example} ${${example}_PARAMETERS}
    )

    if(HPX_WITH_TESTS
       AND HPX_WITH_TESTS_EXAMPLES
       AND NOT ("${example_program}" IN_LIST disabled_tests)
    )
      add_hpx_example_test(
        "modules.program_options" ${example} ${${example}_PARAMETERS}
      )
    endif()

  endforeach()

endif()
