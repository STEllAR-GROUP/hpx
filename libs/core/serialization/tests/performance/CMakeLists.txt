# Copyright (c) 2019 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(benchmarks serialization_performance)
set(serialization_performance_PARAMETERS 100)

if(HPX_WITH_NETWORKING)
  set(benchmarks ${benchmarks} serialization_overhead)
  set(serialization_overhead_FLAGS DEPENDENCIES iostreams_component)
endif()

foreach(benchmark ${benchmarks})
  set(sources ${benchmark}.cpp)

  source_group("Source Files" FILES ${sources})

  set(folder_name "Benchmarks/Modules/Core/Serialization")

  # add example executable
  add_hpx_executable(
    ${benchmark}_test INTERNAL_FLAGS
    SOURCES ${sources} ${${benchmark}_FLAGS}
    EXCLUDE_FROM_ALL
    FOLDER ${folder_name}
  )

  add_hpx_performance_test(
    "modules.serialization" ${benchmark} ${${benchmark}_PARAMETERS}
  )
endforeach()
