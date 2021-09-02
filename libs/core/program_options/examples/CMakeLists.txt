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

if(HPXLocal_WITH_EXAMPLES)
  hpx_local_add_pseudo_target(examples.modules.program_options)
  hpx_local_add_pseudo_dependencies(
    examples.modules examples.modules.program_options
  )

  if(HPXLocal_WITH_TESTS AND HPXLocal_WITH_TESTS_EXAMPLES)
    hpx_local_add_pseudo_target(tests.examples.modules.program_options)
    hpx_local_add_pseudo_dependencies(
      tests.examples.modules tests.examples.modules.program_options
    )
  endif()

  foreach(example ${examples})

    set(sources ${example}.cpp)

    source_group("Source Files" FILES ${sources})

    set(folder_name "Examples/Modules/Core/ProgramOptions")

    # add example executable
    hpx_local_add_executable(
      ${example} INTERNAL_FLAGS
      SOURCES ${sources} ${${example}_FLAGS}
      DEPENDENCIES ${${example}_LIBS}
      FOLDER ${folder_name}
    )

    hpx_local_add_example_target_dependencies(
      "modules.program_options" ${example} ${${example}_PARAMETERS}
    )

    if(HPXLocal_WITH_TESTS
       AND HPXLocal_WITH_TESTS_EXAMPLES
       AND NOT ("${example_program}" IN_LIST disabled_tests)
    )
      hpx_local_add_example_test(
        "modules.program_options" ${example} ${${example}_PARAMETERS}
      )
    endif()

  endforeach()

endif()
