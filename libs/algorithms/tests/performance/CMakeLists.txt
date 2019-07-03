# Copyright (c) 2017 Taeguk Kwon
# Copyright (c) 2017 Google
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(benchmarks
    benchmark_inplace_merge
    benchmark_is_heap
    benchmark_is_heap_until
    benchmark_merge
    benchmark_partition
    benchmark_partition_copy
    benchmark_remove
    benchmark_remove_if
    benchmark_unique
    benchmark_unique_copy
   )

foreach(benchmark ${benchmarks})
  set(sources
      ${benchmark}.cpp)

  source_group("Source Files" FILES ${sources})

  # add example executable
  add_hpx_executable(${benchmark}
                     SOURCES ${sources}
                     EXCLUDE_FROM_ALL
                     ${${benchmark}_FLAGS}
                     FOLDER "Benchmarks/Modules/Algorithms")

  # add a custom target for this example
  add_hpx_pseudo_target(tests.performance.modules.algorithms.${benchmark})

  # make pseudo-targets depend on master pseudo-target
  add_hpx_pseudo_dependencies(tests.performance.modules.algorithms
                        tests.performance.modules.algorithms.${benchmark})

  # add dependencies to pseudo-target
  add_hpx_pseudo_dependencies(tests.performance.modules.algorithms.${benchmark}
                              ${benchmark})
endforeach()
