# Copyright (c) 2019 The STE||AR-Group
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(tests
    is_iterator
    is_range
    range
)

# Set the dependencies of each test
set(is_iterator_LIBS
  DEPENDENCIES hpx_config hpx_iterator_support partitioned_vector_component
    hpx_testing)

set(is_range_LIBS DEPENDENCIES hpx_config hpx_iterator_support hpx_testing)

set(range_LIBS DEPENDENCIES hpx_config hpx_iterator_support hpx_testing)

# Add the tests
foreach(test ${tests})
  set(sources ${test}.cpp)

  source_group("Source Files" FILES ${sources})

  add_hpx_executable(${test}_test
                     SOURCES ${sources}
                     ${${test}_FLAGS}
                     ${${test}_LIBS}
                     EXCLUDE_FROM_ALL
                     HPX_PREFIX ${HPX_BUILD_PREFIX}
                     FOLDER "Tests/Unit/Modules/Iterator_support")

  add_hpx_unit_test("modules.iterator_support" ${test} ${${test}_PARAMETERS})

endforeach()
