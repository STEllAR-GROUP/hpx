# Copyright (c) 2007-2023 Hartmut Kaiser
# Copyright (c)      2014 Thomas Heller
# Copyright (c) 2011-2012 Bryce Adelstein-Lelbach
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

if(NOT HPX_WITH_NETWORKING)
  return()
endif()

set(tests put_parcels set_parcel_write_handler zero_copy_parcel)

set(put_parcels_PARAMETERS LOCALITIES 2)
set(set_parcel_write_handler_PARAMETERS LOCALITIES 2)
set(zero_copy_parcel_PARAMETERS LOCALITIES 2)

foreach(test ${tests})
  set(sources ${test}.cpp)

  source_group("Source Files" FILES ${sources})

  # add example executable
  add_hpx_executable(
    ${test}_test INTERNAL_FLAGS
    SOURCES ${sources} ${${test}_FLAGS}
    EXCLUDE_FROM_ALL
    HPX_PREFIX ${HPX_BUILD_PREFIX}
    FOLDER "Tests/Unit/Modules/Full/Parcelset"
  )

  target_include_directories(${test}_test SYSTEM PRIVATE ${CUDA_INCLUDE_DIRS})
  add_hpx_unit_test(
    "modules.parcelset" ${test} ${${test}_PARAMETERS} RUN_SERIAL
  )

endforeach()

# run zero_copy_parcel with three additional configurations
add_hpx_unit_test(
  "modules.parcelset" zero_copy_parcel_no_array_optimization
  EXECUTABLE zero_copy_parcel
  PSEUDO_DEPS_NAME zero_copy_parcel ${zero_copy_parcel_PARAMETERS}
  RUN_SERIAL
  ARGS --hpx:ini=hpx.parcel.array_optimization=0
)

add_hpx_unit_test(
  "modules.parcelset" zero_copy_parcel_no_zero_copy_optimization
  EXECUTABLE zero_copy_parcel
  PSEUDO_DEPS_NAME zero_copy_parcel ${zero_copy_parcel_PARAMETERS}
  RUN_SERIAL
  ARGS --hpx:ini=hpx.parcel.zero_copy_optimization=0
)

add_hpx_unit_test(
  "modules.parcelset" zero_copy_parcel_no_zero_copy_receive_optimization
  EXECUTABLE zero_copy_parcel
  PSEUDO_DEPS_NAME zero_copy_parcel ${zero_copy_parcel_PARAMETERS}
  RUN_SERIAL
  ARGS --hpx:ini=hpx.parcel.zero_copy_receive_optimization=0
)
