# Copyright (c) 2017 Taeguk Kwon
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(benchmarks
    benchmark_is_heap
    benchmark_is_heap_until
   )

foreach(benchmark ${benchmarks})
  set(sources
      ${benchmark}.cpp)

  source_group("Source Files" FILES ${sources})

  # add example executable
  add_hpx_executable(${benchmark}
                     SOURCES ${sources}
                     ${${benchmark}_FLAGS}
                     FOLDER "Benchmarks/ParallelAlgorithms/Local")

  # add a custom target for this example
  add_hpx_pseudo_target(tests.performance.parallel_algorithms.local.${benchmark})

  # make pseudo-targets depend on master pseudo-target
  add_hpx_pseudo_dependencies(tests.performance.parallel_algorithms.local
                        tests.performance.parallel_algorithms.local.${benchmark})

  # add dependencies to pseudo-target
  add_hpx_pseudo_dependencies(tests.performance.parallel_algorithms.local.${benchmark}
                              ${benchmark}_exe)
endforeach()
