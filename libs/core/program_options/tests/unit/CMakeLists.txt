# Copyright (c) 2019 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(tests
    cmdline
    options_exception
    optional
    options_description
    parsers
    positional_options
    required
    split
    unicode
    unrecognized
    variable_map
    winmain
)

foreach(test ${tests})
  set(sources ${test}.cpp)

  source_group("Source Files" FILES ${sources})

  set(folder_name "Tests/Unit/Modules/Core/ProgramOptions")

  # add example executable
  hpx_local_add_executable(
    ${test}_test INTERNAL_FLAGS
    SOURCES ${sources} ${${test}_FLAGS}
    EXCLUDE_FROM_ALL
    FOLDER ${folder_name}
  )

  hpx_local_add_unit_test(
    "modules.program_options" ${test} ${${test}_PARAMETERS}
  )
endforeach()
