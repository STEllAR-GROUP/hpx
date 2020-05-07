# Copyright (c) 2007-2016 Hartmut Kaiser
# Copyright (c)      2014 Thomas Heller
# Copyright (c) 2011-2012 Bryce Adelstein-Lelbach
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(tests put_parcels set_parcel_write_handler)

set(put_parcels_PARAMETERS LOCALITIES 2)
set(put_parcels_FLAGS DEPENDENCIES iostreams_component)
set(set_parcel_write_handler_PARAMETERS LOCALITIES 2)

if(HPX_WITH_PARCEL_COALESCING)
  set(tests ${tests} put_parcels_with_coalescing)
  set(put_parcels_with_coalescing_PARAMETERS LOCALITIES 2)
  set(put_parcels_with_coalescing_FLAGS DEPENDENCIES iostreams_component
                                        parcel_coalescing
  )
endif()

if(HPX_WITH_COMPRESSION_BZIP2
   OR HPX_WITH_COMPRESSION_ZLIB
   OR HPX_WITH_COMPRESSION_SNAPPY
)
  set(tests ${tests} put_parcels_with_compression)
  set(put_parcels_with_compression_PARAMETERS LOCALITIES 2)
  set(put_parcels_with_compression_FLAGS DEPENDENCIES iostreams_component)
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
    FOLDER "Tests/Unit/Parcelset"
  )

  add_hpx_unit_test("parcelset" ${test} ${${test}_PARAMETERS})

endforeach()
