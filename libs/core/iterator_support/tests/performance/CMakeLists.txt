# Copyright (c) 2019 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(benchmarks)

if(HPX_WITH_DISTRIBUTED_RUNTIME)
  set(benchmarks ${benchmarks} stencil3_iterators)
  set(stencil3_iterators_FLAGS DEPENDENCIES iostreams_component hpx_timing)
endif()

foreach(benchmark ${benchmarks})
  set(sources ${benchmark}.cpp)

  source_group("Source Files" FILES ${sources})

  # add example executable
  add_hpx_executable(
    ${benchmark}_test INTERNAL_FLAGS
    SOURCES ${sources}
    EXCLUDE_FROM_ALL ${${benchmark}_FLAGS}
    DEPENDENCIES iostreams_component hpx_iterator_support hpx_timing
    FOLDER "Benchmarks/Modules/Core/IteratorSupport"
  )

  add_hpx_performance_test(
    "modules.iterator_support" ${benchmark} ${${benchmark}_PARAMETERS}
  )

endforeach()
