# Copyright (c) 2007-2012 Hartmut Kaiser
# Copyright (c) 2011 Bryce Adelstein-Lelbach
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(tests)

if(HPX_WITH_DISTRIBUTED_RUNTIME)
  set(tests ${tests} function_argument function_serialization_728)
endif()

set(function_argument_FLAGS DEPENDENCIES iostreams_component)
set(function_serialization_728_dependencies iostreams_component)

if(HPX_WITH_COMPRESSION_ZLIB)
  set(function_serialization_728_dependencies
      ${function_serialization_728_dependencies} compression_zlib
  )
endif()
if(WITH_PARCEL_COALESCING)
  set(function_serialization_728_dependencies
      ${function_serialization_728_dependencies} parcel_coalescing
  )
endif()
set(function_serialization_728_FLAGS DEPENDENCIES
                                     ${function_serialization_728_dependencies}
)

foreach(test ${tests})
  set(sources ${test}.cpp)

  source_group("Source Files" FILES ${sources})

  add_hpx_executable(
    ${test}_test INTERNAL_FLAGS
    SOURCES ${sources} ${${test}_FLAGS}
    EXCLUDE_FROM_ALL
    HPX_PREFIX ${HPX_BUILD_PREFIX}
    FOLDER "Tests/Regressions/Modules/Core/Functional"
  )

  add_hpx_regression_test("modules.functional" ${test} ${${test}_PARAMETERS})
endforeach()
