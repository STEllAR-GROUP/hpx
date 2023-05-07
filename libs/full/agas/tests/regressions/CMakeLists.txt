# Copyright (c) 2020-2022 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(tests duplicate_id_registration_1596 id_type_ref_counting_1032
          pass_by_value_id_type_action send_gid_keep_component_1624
)

if(HPX_WITH_NETWORKING)
  set(tests ${tests} register_with_basename_1804)
  set(register_with_basename_1804_PARAMETERS LOCALITIES 2)
endif()

set(id_type_ref_counting_1032_PARAMETERS THREADS_PER_LOCALITY 1)

foreach(test ${tests})
  set(sources ${test}.cpp)

  source_group("Source Files" FILES ${sources})

  # add example executable
  add_hpx_executable(
    ${test}_test INTERNAL_FLAGS
    SOURCES ${sources} ${${test}_FLAGS}
    EXCLUDE_FROM_ALL
    HPX_PREFIX ${HPX_BUILD_PREFIX}
    FOLDER "Tests/Regressions/Modules/Full/AGAS"
  )

  add_hpx_regression_test("modules.agas" ${test} ${${test}_PARAMETERS})
endforeach()

add_hpx_regression_test(
  "" id_type_ref_counting_1032_4
  EXECUTABLE id_type_ref_counting_1032
  PSEUDO_DEPS_NAME id_type_ref_counting_1032
  THREADS_PER_LOCALITY 4
)
