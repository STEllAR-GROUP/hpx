#  Copyright (c) 2014-2016 Hartmut Kaiser
#  Copyright (c) 2011 Thomas Heller
#
# SPDX-License-Identifier: BSL-1.0
#  Distributed under the Boost Software License, Version 1.0. (See accompanying
#  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(tests)

if(HPX_WITH_DISTRIBUTED_RUNTIME)
  set(tests migrate_component_to_storage new_binpacking new_colocated)

  set(migrate_component_to_storage_FLAGS DEPENDENCIES unordered_component
                                         component_storage_component
  )

  set(new_binpacking_PARAMETERS LOCALITIES 2)
  set(new_colocated_PARAMETERS LOCALITIES 2)
endif()

foreach(test ${tests})
  set(sources ${test}.cpp)

  source_group("Source Files" FILES ${sources})

  set(folder_name "Tests/Unit/Components")

  # add example executable
  add_hpx_executable(
    ${test}_test INTERNAL_FLAGS
    SOURCES ${sources} ${${test}_FLAGS}
    EXCLUDE_FROM_ALL
    HPX_PREFIX ${HPX_BUILD_PREFIX}
    FOLDER ${folder_name}
  )

  add_hpx_unit_test("component" ${test} ${${test}_PARAMETERS} RUN_SERIAL)

endforeach()
