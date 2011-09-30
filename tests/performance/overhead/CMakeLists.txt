# Copyright (c) 2011 Bryce Lelbach
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying 
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(benchmarks
    delay_baseline_overhead
   )

set(delay_baseline_overhead_FLAGS DEPENDENCIES iostreams_component)

if(HPX_CXX11_LAMBDAS)
  set(benchmarks ${benchmarks}
      eager_future_overhead
     )

  set(eager_future_overhead_FLAGS DEPENDENCIES iostreams_component)
endif()

foreach(benchmark ${benchmarks})
  set(sources
      ${benchmark}.cpp)

  source_group("Source Files" FILES ${sources})

  # add example executable
  add_hpx_executable(${benchmark}
                     MODULE overhead
                     SOURCES ${sources}
                     ${${benchmark}_FLAGS})

  # add a custom target for this example 
  add_hpx_pseudo_target(tests.performance.overhead.${benchmark})

  # make pseudo-targets depend on master pseudo-target
  add_hpx_pseudo_dependencies(tests.performance.overhead
                              tests.performance.overhead.${benchmark})

  # add dependencies to pseudo-target 
  add_hpx_pseudo_dependencies(tests.performance.overhead.${benchmark}
                              ${benchmark}_exe)
endforeach() 
 
