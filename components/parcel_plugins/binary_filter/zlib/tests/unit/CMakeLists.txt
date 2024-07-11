# Copyright (c) 2019-2022 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(tests put_parcels_with_compression_zlib)

set(put_parcels_with_compression_zlib_PARAMETERS LOCALITIES 2)
set(put_parcels_with_compression_zlib_FLAGS DEPENDENCIES compression_zlib)

foreach(test ${tests})
  set(sources ${test}.cpp)

  source_group("Source Files" FILES ${sources})

  # add example executable
  add_hpx_executable(
    ${test}_test INTERNAL_FLAGS
    SOURCES ${sources} ${${test}_FLAGS}
    EXCLUDE_FROM_ALL
    HPX_PREFIX ${HPX_BUILD_PREFIX}
    FOLDER "Tests/Unit/Full/Plugins/Compression"
  )

  add_hpx_unit_test(
    "components.parcel_plugins.binary_filter.zlib" ${test}
    ${${test}_PARAMETERS}
  )
endforeach()
