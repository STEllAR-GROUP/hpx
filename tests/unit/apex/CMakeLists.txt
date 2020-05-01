#  Copyright (c) 2015 University of Oregon
#
# SPDX-License-Identifier: BSL-1.0
#  Distributed under the Boost Software License, Version 1.0. (See accompanying
#  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(tests apex_action_count annotation_check)

foreach(test ${tests})
  set(sources ${test}.cpp)

  source_group("Source Files" FILES ${sources})

  # add example executable
  add_hpx_executable(
    ${test}_test INTERNAL_FLAGS
    SOURCES ${sources} ${${test}_FLAGS}
    EXCLUDE_FROM_ALL
    HPX_PREFIX ${HPX_BUILD_PREFIX}
    FOLDER "Tests/Unit/Apex"
  )

  add_hpx_unit_test("apex" ${test} ${${test}_PARAMETERS})

endforeach()

# dataflow support works for the following (add sync - if it ever works)
foreach(
  launch
  "" "Executor non-threaded "
  # "Executor threaded "
  "apply " "async " "deferred " "fork "
)
  set(REGEX_MATCH_D_ "${REGEX_MATCH_D_}.*1-${launch}Dataflow")
endforeach()

# continuation support works for the following
foreach(launch "" "Executor non-threaded " # Executor threaded "
               "async " "fork "
)
  set(REGEX_MATCH_C_ "${REGEX_MATCH_C_}.*2-${launch}Continuation")
endforeach()

set_tests_properties(
  tests.unit.apex.annotation_check
  PROPERTIES PASS_REGULAR_EXPRESSION "${REGEX_MATCH_D_}${REGEX_MATCH_C_}"
)
