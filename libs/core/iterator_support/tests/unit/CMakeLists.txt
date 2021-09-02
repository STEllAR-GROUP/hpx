# Copyright (c) 2019-2021 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(tests
    boost_iterator_categories
    counting_iterator
    is_iterator
    is_range
    iterator_adaptor
    iterator_facade
    range
    stencil3_iterator
    transform_iterator
    transform_iterator2
    zip_iterator
    is_sentinel_for
    is_sized_sentinel_for
)

set(is_range_FLAGS NOLIBS)
set(is_range_LIBS DEPENDENCIES hpx_local)

set(range_FLAGS NOLIBS)
set(range_LIBS DEPENDENCIES hpx_local)

# Add the tests
foreach(test ${tests})
  set(sources ${test}.cpp)

  source_group("Source Files" FILES ${sources})

  hpx_local_add_executable(
    ${test}_test INTERNAL_FLAGS
    SOURCES ${sources} ${${test}_FLAGS} ${${test}_LIBS}
    EXCLUDE_FROM_ALL
    HPX_PREFIX ${HPX_BUILD_PREFIX}
    FOLDER "Tests/Unit/Modules/Core/IteratorSupport"
  )

  target_link_libraries(
    ${test}_test PRIVATE hpx_iterator_support_test_utilities
  )

  hpx_local_add_unit_test(
    "modules.iterator_support" ${test} ${${test}_PARAMETERS}
  )
endforeach()
