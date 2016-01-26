# Copyright (c) 2016 Hartmut Kaiser
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(benchmarks
    minmax_element_performance)

foreach(benchmark ${benchmarks})

  set(sources
      ${benchmark}.cpp)

  source_group("Source Files" FILES ${sources})

  # add example executable
  add_hpx_executable(${benchmark}
                     SOURCES ${sources}
                     ${${benchmark}_FLAGS}
                     COMPONENT_DEPENDENCIES iostreams
                     EXCLUDE_FROM_ALL
                     HPX_PREFIX ${HPX_BUILD_PREFIX}
                     FOLDER "Benchmarks/Network/Algorithms/${benchmark}")

  # add a custom target for this example
  add_hpx_pseudo_target(tests.performance.network.algorithms.${benchmark})

  # make pseudo-targets depend on master pseudo-target
  add_hpx_pseudo_dependencies(tests.performance.network.algorithms
                              tests.performance.network.algorithms.${benchmark})

  # add dependencies to pseudo-target
  add_hpx_pseudo_dependencies(tests.performance.network.algorithms.${benchmark}
                              ${benchmark}_exe)
endforeach()

