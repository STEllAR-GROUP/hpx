# Copyright (c) 2011 Bryce Adelstein-Lelbach
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(benchmarks
    boost_tls_overhead
    native_tls_overhead
    hpx_tls_overhead
    delay_baseline
    hpx_homogeneous_task_spawn
    hpx_heterogeneous_task_spawn
    print_heterogeneous_payloads
    sizeof
   )

if(OPENMP_FOUND)
  set(benchmarks
      ${benchmarks}
      openmp_homogeneous_task_spawn
     )

  set(openmp_homogeneous_task_spawn_FLAGS NOLIBS
      DEPENDENCIES ${BOOST_program_options_LIBRARY})
endif()

if(QTHREADS_FOUND)
  include_directories(${QTHREADS_INCLUDE_DIR})

  set(benchmarks
      ${benchmarks}
      qthreads_homogeneous_task_spawn
      qthreads_heterogeneous_task_spawn
     )

  set(qthreads_homogeneous_task_spawn_FLAGS NOLIBS
      DEPENDENCIES ${BOOST_program_options_LIBRARY}
                   ${QTHREADS_LIBRARY})

  set(qthreads_heterogeneous_task_spawn_FLAGS NOLIBS
      DEPENDENCIES ${BOOST_program_options_LIBRARY}
                   ${QTHREADS_LIBRARY})
endif()

if(TBB_FOUND)
  include_directories(${TBB_INCLUDE_DIR})

  set(benchmarks
      ${benchmarks}
      tbb_homogeneous_task_spawn
     )

  set(tbb_homogeneous_task_spawn_FLAGS NOLIBS
      DEPENDENCIES ${BOOST_program_options_LIBRARY}
                   ${TBB_LIBRARY})
endif()

if(SWARM_FOUND)
  include_directories(${SWARM_INCLUDE_DIR})

  set(benchmarks
      ${benchmarks}
      swarm_homogeneous_task_spawn
     )

  set(swarm_homogeneous_task_spawn_FLAGS NOLIBS
      DEPENDENCIES ${BOOST_program_options_LIBRARY}
                   ${SWARM_LIBRARY})
endif()

set(boost_tls_overhead_FLAGS NOLIBS
    DEPENDENCIES ${BOOST_program_options_LIBRARY}
                 ${BOOST_thread_LIBRARY})

set(native_tls_overhead_FLAGS NOLIBS
    DEPENDENCIES ${BOOST_program_options_LIBRARY}
                 ${BOOST_thread_LIBRARY})

set(hpx_tls_overhead_FLAGS NOLIBS
    DEPENDENCIES ${BOOST_program_options_LIBRARY}
                 ${BOOST_thread_LIBRARY})

set(delay_baseline_FLAGS NOLIBS
    DEPENDENCIES ${BOOST_program_options_LIBRARY})

set(hpx_homogeneous_task_spawn_FLAGS DEPENDENCIES iostreams_component)

set(hpx_heterogeneous_task_spawn_FLAGS DEPENDENCIES iostreams_component)

set(print_heterogeneous_payloads_FLAGS NOLIBS
    DEPENDENCIES ${BOOST_program_options_LIBRARY})

if(HPX_CXX11_LAMBDAS)
  set(benchmarks ${benchmarks}
      packaged_task_overhead
      spinlock_overhead1
      spinlock_overhead2
     )

  set(packaged_task_overhead_FLAGS DEPENDENCIES iostreams_component)

  set(spinlock_overhead1_FLAGS DEPENDENCIES iostreams_component)

  set(spinlock_overhead2_FLAGS DEPENDENCIES iostreams_component)
endif()

set(sizeof_FLAGS DEPENDENCIES iostreams_component)

foreach(benchmark ${benchmarks})
  set(sources
      ${benchmark}.cpp)

  source_group("Source Files" FILES ${sources})

  # add example executable
  add_hpx_executable(${benchmark}
                     SOURCES ${sources}
                     ${${benchmark}_FLAGS}
                     FOLDER "Benchmarks/${benchmark}")

  # add a custom target for this example
  add_hpx_pseudo_target(tests.performance.${benchmark})

  # make pseudo-targets depend on master pseudo-target
  add_hpx_pseudo_dependencies(tests.performance
                              tests.performance.${benchmark})

  # add dependencies to pseudo-target
  add_hpx_pseudo_dependencies(tests.performance.${benchmark}
                              ${benchmark}_exe)
endforeach()

if(OPENMP_FOUND)
  set_target_properties(openmp_homogeneous_task_spawn_exe PROPERTIES COMPILE_FLAGS ${OpenMP_CXX_FLAGS})
  set_target_properties(openmp_homogeneous_task_spawn_exe PROPERTIES LINK_FLAGS ${OpenMP_CXX_FLAGS})
endif()

add_hpx_executable(util_function
                   SOURCES util/function.cpp
                   ${util_function_FLAGS}
                   FOLDER "Benchmarks/util_function")

add_hpx_pseudo_target(tests.performance.util.function)

add_hpx_pseudo_dependencies(tests.performance
                            tests.performance.util.function)

add_hpx_pseudo_dependencies(tests.performance.util.function
                            util.function_exe)

