# Copyright (c) 2022 Hartmut Kaiser
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(tests function_serialization_728_zlib)

set(function_serialization_728_zlib_FLAGS DEPENDENCIES compression_zlib)

foreach(test ${tests})
  set(sources ${test}.cpp)

  source_group("Source Files" FILES ${sources})

  # add example executable
  add_hpx_executable(
    ${test}_test INTERNAL_FLAGS
    SOURCES ${sources} ${${test}_FLAGS}
    EXCLUDE_FROM_ALL
    HPX_PREFIX ${HPX_BUILD_PREFIX}
    FOLDER "Tests/Regressions/Full/Plugins/Compression"
  )

  add_hpx_regression_test(
    "components.parcel_plugins.binary_filter.zlib" ${test}
    ${${test}_PARAMETERS}
  )
endforeach()
