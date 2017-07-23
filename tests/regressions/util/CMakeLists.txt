# Copyright (c) 2007-2012 Hartmut Kaiser
# Copyright (c) 2011 Bryce Adelstein-Lelbach
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(tests
    command_line_arguments_706
    configuration_1572
    function_argument
    function_serialization_728
    iarchive_1237
    protect_with_nullary_pfo
    serialize_buffer_1069
    set_config_entry_deadlock
    tuple_serialization_803
    unwrapped_1528
    use_all_cores_2262
    zero_copy_parcels_1001
   )

set(function_argument_FLAGS DEPENDENCIES iostreams_component)
set(function_serialization_728_dependencies iostreams_component)
set(iarchive_1237_FLAGS DEPENDENCIES iostreams_component)
set(serialize_buffer_1069_FLAGS DEPENDENCIES iostreams_component)

if(HPX_WITH_COMPRESSION_ZLIB)
  set(function_serialization_728_dependencies
    ${function_serialization_728_dependencies}
    compress_zlib_lib)
endif()
if(WITH_PARCEL_COALESCING)
  set(function_serialization_728_dependencies
    ${function_serialization_728_dependencies}
    parcel_coalescing_lib)
endif()
set(function_serialization_728_FLAGS DEPENDENCIES ${function_serialization_728_dependencies})

set(zero_copy_parcels_1001_PARAMETERS LOCALITIES 2)

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
                     FOLDER "Tests/Regressions/Util")

  add_hpx_regression_test("util" ${test} ${${test}_PARAMETERS})

  # add a custom target for this example
  add_hpx_pseudo_target(tests.regressions.util_dir.${test})

  # make pseudo-targets depend on master pseudo-target
  add_hpx_pseudo_dependencies(tests.regressions.util_dir
                              tests.regressions.util_dir.${test})

  # add dependencies to pseudo-target
  add_hpx_pseudo_dependencies(tests.regressions.util_dir.${test}
                              ${test}_test_exe)
endforeach()

# run zero_copy_parcels_1001 with two additional configurations
add_hpx_regression_test(
    "util" zero_copy_parcels_1001_no_array_optimization
    EXECUTABLE zero_copy_parcels_1001
    ${zero_copy_parcels_1001_PARAMETERS}
    ARGS --hpx:ini=hpx.parcel.array_optimization=0)

add_hpx_regression_test(
    "util" zero_copy_parcels_1001_no_zero_copy_optimization
    EXECUTABLE zero_copy_parcels_1001
    ${zero_copy_parcels_1001_PARAMETERS}
    ARGS --hpx:ini=hpx.parcel.zero_copy_optimization=0)
