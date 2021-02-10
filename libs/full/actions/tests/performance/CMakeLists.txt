# Copyright (c) 2020 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(benchmarks)

if(HPX_WITH_NETWORKING)
  set(benchmarks ${benchmarks} serialization_overhead)
  set(serialization_overhead_FLAGS DEPENDENCIES iostreams_component)
endif()

foreach(benchmark ${benchmarks})
  set(sources ${benchmark}.cpp)

  source_group("Source Files" FILES ${sources})

  set(folder_name "Benchmarks/Modules/Full/Actions")

  # add example executable
  add_hpx_executable(
    ${benchmark}_test INTERNAL_FLAGS
    SOURCES ${sources} ${${benchmark}_FLAGS}
    EXCLUDE_FROM_ALL
    FOLDER ${folder_name}
  )

  add_hpx_performance_test(
    "modules.actions" ${benchmark} ${${benchmark}_PARAMETERS}
  )
endforeach()
