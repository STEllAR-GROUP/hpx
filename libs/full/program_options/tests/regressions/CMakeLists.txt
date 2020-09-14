# Copyright (c) 2019 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(tests command_line_arguments_706 commandline_options_1437
          command_line_required_arguments_2990
)

set(commandline_options_1437_PARAMETERS THREADS_PER_LOCALITY 2)

foreach(test ${tests})
  set(sources ${test}.cpp)

  source_group("Source Files" FILES ${sources})

  set(folder_name "Tests/Regressions/Modules/Full/ProgramOptions")

  # add example executable
  add_hpx_executable(
    ${test}_test INTERNAL_FLAGS
    SOURCES ${sources} ${${test}_FLAGS}
    EXCLUDE_FROM_ALL
    FOLDER ${folder_name}
  )

  add_hpx_regression_test(
    "modules.program_options" ${test} ${${test}_PARAMETERS}
  )
endforeach()
