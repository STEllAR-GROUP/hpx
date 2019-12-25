# Copyright (c) 2019 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

if (HPX_WITH_EXAMPLES)
  add_hpx_pseudo_target(examples.modules.lexical_cast)
  add_hpx_pseudo_dependencies(examples.modules examples.modules.lexical_cast)
  if (HPX_WITH_TESTS AND HPX_WITH_TESTS_EXAMPLES AND HPX_LEXICAL_CAST_WITH_TESTS)
    add_hpx_pseudo_target(tests.examples.modules.lexical_cast)
    add_hpx_pseudo_dependencies(tests.examples.modules tests.examples.modules.lexical_cast)
  endif()
else()
  return()
endif()

set(example_programs
    args_to_numbers
    generic_stringize
    small_examples
    variant_to_long_double
  )

foreach(example_program ${example_programs})
  set(sources
      ${example_program}.cpp)

  source_group("Source Files" FILES ${sources})

  # add example executable
  add_hpx_executable(${example_program}
    INTERNAL_FLAGS
    SOURCES ${sources}
    ${${example_program}_FLAGS}
    FOLDER "Examples/Modules/LexicalCast")

  add_hpx_example_target_dependencies("modules.lexical_cast" ${example_program})

  if(HPX_WITH_TESTS AND HPX_WITH_TESTS_EXAMPLES)
    add_hpx_example_test("modules.lexical_cast" ${example_program} ${${example_program}_PARAMETERS})
  endif()
endforeach()
