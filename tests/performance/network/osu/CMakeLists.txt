# Copyright (c) 2011 Bryce Adelstein-Lelbach
# Copyright (c) 2012 Daniel Kogler
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(benchmarks
    osu_bibw
    osu_bw
    osu_latency
    osu_multi_lat)

foreach(benchmark ${benchmarks})
  set(sources
      osu_base.cpp
      ${benchmark}.cpp)

  source_group("Source Files" FILES ${sources})

  # add example executable
  add_hpx_executable(${benchmark}
                     SOURCES ${sources}
                     ${${benchmark}_FLAGS}
                     COMPONENT_DEPENDENCIES iostreams
                     EXCLUDE_FROM_ALL
                     HPX_PREFIX ${HPX_BUILD_PREFIX}
                     FOLDER "Benchmarks/Network/${benchmark}")

  # add a custom target for this example
  add_hpx_pseudo_target(tests.performance.network.osu_perf.${benchmark})

  # make pseudo-targets depend on master pseudo-target
  add_hpx_pseudo_dependencies(tests.performance.network.osu_perf
                              tests.performance.network.osu_perf.${benchmark})

  # add dependencies to pseudo-target
  add_hpx_pseudo_dependencies(tests.performance.network.osu_perf.${benchmark}
                              ${benchmark}_exe)
endforeach()


set(coll_benchmarks
    #osu_bcast
    #osu_scatter
    )

foreach(benchmark ${coll_benchmarks})
  set(sources
      ${benchmark}.cpp)

  source_group("Source Files" FILES ${sources})

  # add example executable
  add_hpx_executable(${benchmark}
                     SOURCES ${sources}
                     ${${benchmark}_FLAGS}
                     COMPONENT_DEPENDENCIES iostreams
                     EXCLUDE_FROM_ALL
                     FOLDER "Benchmarks/Network/${benchmark}")

  # add a custom target for this example
  add_hpx_pseudo_target(tests.performance.network.osu_perf.${benchmark})

  # make pseudo-targets depend on master pseudo-target
  add_hpx_pseudo_dependencies(tests.performance.network.osu_perf
                              tests.performance.network.osu_perf.${benchmark})

  # add dependencies to pseudo-target
  add_hpx_pseudo_dependencies(tests.performance.network.osu_perf.${benchmark}
                              ${benchmark}_exe)
endforeach()

