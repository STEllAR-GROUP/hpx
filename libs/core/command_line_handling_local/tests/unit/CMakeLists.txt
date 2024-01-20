# Copyright (c) 2020-2024 The STE||AR-Group
#               2011 Bryce Adelstein-Lelbach
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(tests)

if(HPX_COMMAND_LINE_HANDLING_LOCAL_WITH_JSON_CONFIGURATION_FILES)
  set(tests json_config_file)

  set(json_config_file_PARAMETERS
      "--hpx:options-file=${CMAKE_CURRENT_SOURCE_DIR}/config_file.json"
  )
endif()

foreach(test ${tests})

  set(sources ${test}.cpp)

  source_group("Source Files" FILES ${sources})

  add_hpx_executable(
    ${test}_test INTERNAL_FLAGS
    SOURCES ${sources}
    EXCLUDE_FROM_ALL
    FOLDER "Tests/Unit/Modules/Core/CommandLineHandlingLocal/"
  )

  add_hpx_unit_test(
    "modules.command_line_handling_local" ${test} ${${test}_PARAMETERS}
  )

endforeach()
