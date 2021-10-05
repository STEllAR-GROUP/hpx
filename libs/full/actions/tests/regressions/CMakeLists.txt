# Copyright (c) 2019 The STE||AR-Group
# Copyright (c) 2007-2012 Hartmut Kaiser
# Copyright (c) 2011 Bryce Adelstein-Lelbach
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(tests tuple_serialization_803 function_argument function_serialization_728
          wait_all_hang_1946
)

set(function_argument_FLAGS DEPENDENCIES iostreams_component)
set(function_serialization_728_dependencies iostreams_component)
set(wait_all_hang_1946_PARAMETERS THREADS_PER_LOCALITY 8)
set(wait_all_hang_1946_FLAGS DEPENDENCIES iostreams_component)

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

  set(folder_name "Tests/Regressions/Modules/Full/Actions")

  # add example executable
  add_hpx_executable(
    ${test}_test INTERNAL_FLAGS
    SOURCES ${sources} ${${test}_FLAGS}
    EXCLUDE_FROM_ALL
    FOLDER ${folder_name}
  )

  add_hpx_regression_test("modules.actions" ${test} ${${test}_PARAMETERS})
endforeach()
