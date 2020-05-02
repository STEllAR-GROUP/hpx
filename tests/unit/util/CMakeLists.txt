# Copyright (c) 2011 Bryce Adelstein-Lelbach
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(tests config_entry)

set(subdirs function)

if(HPX_WITH_DISTRIBUTED_RUNTIME)
  set(tests ${tests} bind_action)
endif()

# ##############################################################################
set(serialize_buffer_PARAMETERS LOCALITIES 2 THREADS_PER_LOCALITY 2)

foreach(test ${tests})
  set(sources ${test}.cpp)

  source_group("Source Files" FILES ${sources})

  # add example executable
  add_hpx_executable(
    ${test}_test INTERNAL_FLAGS
    SOURCES ${sources} ${${test}_FLAGS}
    EXCLUDE_FROM_ALL
    HPX_PREFIX ${HPX_BUILD_PREFIX}
    FOLDER "Tests/Unit/Util/"
  )

  add_hpx_unit_test("util" ${test} ${${test}_PARAMETERS})

endforeach()

# add sub directories
foreach(subdir ${subdirs})
  add_hpx_pseudo_target(tests.unit.util.${subdir})
  add_subdirectory(${subdir})
  add_hpx_pseudo_dependencies(tests.unit.util tests.unit.util.${subdir})
endforeach()
