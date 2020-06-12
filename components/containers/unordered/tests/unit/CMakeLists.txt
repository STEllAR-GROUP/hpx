#  Copyright (c) 2014-2016 Hartmut Kaiser
#  Copyright (c) 2011 Thomas Heller
#
# SPDX-License-Identifier: BSL-1.0
#  Distributed under the Boost Software License, Version 1.0. (See accompanying
#  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(tests unordered_map)

set(unordered_map_FLAGS COMPONENT_DEPENDENCIES unordered)

set(unordered_map_PARAMETERS LOCALITIES 2 THREADS_PER_LOCALITY 2)

foreach(test ${tests})
  set(sources ${test}.cpp)

  source_group("Source Files" FILES ${sources})

  set(folder_name "Tests/Unit/Components/Containers/Unordered")

  # add example executable
  add_hpx_executable(
    ${test}_test INTERNAL_FLAGS
    SOURCES ${sources} ${${test}_FLAGS}
    EXCLUDE_FROM_ALL
    HPX_PREFIX ${HPX_BUILD_PREFIX}
    FOLDER ${folder_name}
  )

  add_hpx_unit_test("components.unordered" ${test} ${${test}_PARAMETERS})
endforeach()
