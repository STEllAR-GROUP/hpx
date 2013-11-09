# Copyright (c) 2007-2012 Hartmut Kaiser
# Copyright (c) 2011 Bryce Adelstein-Lelbach
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(tests
    command_line_arguments_706
    function_argument
    function_serialization_728
    protect_with_nullary_pfo
    tuple_serialization_803
    zero_copy_parcels_1001
   )

set(function_argument_FLAGS DEPENDENCIES iostreams_component)

set(function_serialization_728_dependencies
  iostreams_component)

if(HPX_HAVE_COMPRESSION_ZLIB AND ZLIB_FOUND)
  set(function_serialization_728_dependencies
    ${function_serialization_728_dependencies}
    compress_zlib_lib)
endif()
if(HPX_USE_PARCEL_COALESCING)
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
                     FOLDER "Tests/Regressions/Util")

  add_hpx_regression_test("util" ${test} ${${test}_PARAMETERS})

  # add a custom target for this example
  add_hpx_pseudo_target(tests.regressions.util.${test})

  # make pseudo-targets depend on master pseudo-target
  add_hpx_pseudo_dependencies(tests.regressions.util
                              tests.regressions.util.${test})

  # add dependencies to pseudo-target
  add_hpx_pseudo_dependencies(tests.regressions.util.${test}
                              ${test}_test_exe)
endforeach()

