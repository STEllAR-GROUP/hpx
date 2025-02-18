# Copyright (c) 2024 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

# Copyright (c) 2024 Hartmut Kaiser
#
# SPDX-License-Identifier: BSL-1.0 Distributed under the Boost Software License,
# Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at
# http://www.boost.org/LICENSE_1_0.txt)

if(HPX_WITH_NETWORKING)
  set(tests ${tests} very_big_parcel)
  set(tests ${tests} very_big_tchunk)
  set(very_big_parcel_PARAMETERS LOCALITIES 2)
  set(very_big_tchunk_PARAMETERS LOCALITIES 2)
endif()

foreach(test ${tests})
  set(sources ${test}.cpp)

  source_group("Source Files" FILES ${sources})

  # add example executable
  add_hpx_executable(
    ${test}_test INTERNAL_FLAGS
    SOURCES ${sources} ${${test}_FLAGS}
    EXCLUDE_FROM_ALL
    HPX_PREFIX ${HPX_BUILD_PREFIX}
    FOLDER "Tests/Regressions/Modules/Full/Parcelset"
  )

  # Disable the test for very_big_parcel due to limited CircleCI resources
  if(${test} STREQUAL "very_big_parcel")
    continue()
  endif()
  add_hpx_regression_test(
    "modules.parcelset" ${test} ${${test}_PARAMETERS} TIMEOUT 900
  )

endforeach()

if(HPX_WITH_NETWORKING)
  # Disable the test due to limited CircleCI resources

  # very_big_parcel with one additional configurations

  # add_hpx_regression_test( "modules.parcelset" very_big_parcel_int_max_plus_1
  # EXECUTABLE very_big_parcel TIMEOUT 900 PSEUDO_DEPS_NAME very_big_parcel
  # ${very_big_parcel_PARAMETERS} --nbytes-add=1 )
endif()
