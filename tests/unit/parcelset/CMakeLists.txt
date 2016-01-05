# Copyright (c) 2007-2016 Hartmut Kaiser
# Copyright (c)      2014 Thomas Heller
# Copyright (c) 2011-2012 Bryce Adelstein-Lelbach
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(tests
  put_parcels
  set_parcel_write_handler
)

set(put_parcels_PARAMETERS LOCALITIES 2)
set(set_parcel_write_handler_PARAMETERS LOCALITIES 2)

if(HPX_WITH_PARCEL_COALESCING)
  set(tests ${tests} put_parcels_with_coalescing)
  set(put_parcels_with_coalescing_PARAMETERS LOCALITIES 2)
endif()

foreach(test ${tests})
  set(sources
      ${test}.cpp)

  source_group("Source Files" FILES ${sources})

  # add example executable
  add_hpx_executable(${test}_test
                     SOURCES ${sources}
                     ${${test}_FLAGS}
                     EXCLUDE_FROM_ALL
                     HPX_PREFIX ${HPX_BUILD_PREFIX}
                     FOLDER "Tests/Unit/Parcelset")

  add_hpx_unit_test("parcelset" ${test} ${${test}_PARAMETERS})

  # add a custom target for this example
  add_hpx_pseudo_target(tests.unit.parcelset.${test})

  # make pseudo-targets depend on master pseudo-target
  add_hpx_pseudo_dependencies(tests.unit.parcelset
                              tests.unit.parcelset.${test})

  # add dependencies to pseudo-target
  add_hpx_pseudo_dependencies(tests.unit.parcelset.${test}
                              ${test}_test_exe)
endforeach()
