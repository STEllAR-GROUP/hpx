# Copyright (c) 2011 Bryce Adelstein-Lelbach
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

################################################################################
# find dependencies
################################################################################
find_package(HPX_Qthreads)
find_package(HPX_TBB)
find_package(HPX_SWARM)
find_package(HPX_OpenMP)

if(MSVC)
  set(boost_library_dependencies)
else()
  set(boost_library_dependencies "${BOOST_PROGRAM_OPTIONS_LIBRARY}"
                                 "${BOOST_SYSTEM_LIBRARY}"
                                 "${BOOST_CHRONO_LIBRARY}")
endif()

set(benchmarks
    timed_task_spawn
    hpx_homogeneous_timed_task_spawn_executors
    hpx_heterogeneous_timed_task_spawn

    delay_baseline
    delay_baseline_threaded
    print_heterogeneous_payloads
   )

if(OPENMP_FOUND)
  set(benchmarks
      ${benchmarks}
      openmp_homogeneous_timed_task_spawn
     )

  set(openmp_homogeneous_timed_task_spawn_FLAGS NOLIBS
      DEPENDENCIES ${boost_library_dependencies})
endif()

if(QTHREADS_FOUND)
  include_directories("${QTHREADS_INCLUDE_DIR}")

  set(benchmarks
      ${benchmarks}
      qthreads_homogeneous_timed_task_spawn
      qthreads_heterogeneous_timed_task_spawn
     )

  set(qthreads_homogeneous_timed_task_spawn_FLAGS NOLIBS
      DEPENDENCIES ${boost_library_dependencies}
                   "${QTHREADS_LIBRARY}")

  set(qthreads_heterogeneous_timed_task_spawn_FLAGS NOLIBS
      DEPENDENCIES ${boost_library_dependencies}
                   "${QTHREADS_LIBRARY}")
endif()

if(TBB_FOUND)
  include_directories("${TBB_INCLUDE_DIR}")

  set(benchmarks
      ${benchmarks}
      tbb_homogeneous_timed_task_spawn
     )

  set(tbb_homogeneous_timed_task_spawn_FLAGS NOLIBS
      DEPENDENCIES ${boost_library_dependencies}
                   "${TBB_LIBRARY}")
endif()

if(SWARM_FOUND)
  include_directories("${SWARM_INCLUDE_DIR}")

  set(benchmarks
      ${benchmarks}
      swarm_homogeneous_timed_task_spawn
     )

  set(swarm_homogeneous_timed_task_spawn_FLAGS NOLIBS
      DEPENDENCIES ${boost_library_dependencies}
                   "${SWARM_LIBRARY}")
endif()

set(hpx_homogeneous_timed_task_spawn_executors_FLAGS DEPENDENCIES iostreams_component)
set(hpx_heterogeneous_timed_task_spawn_FLAGS DEPENDENCIES iostreams_component)

set(delay_baseline_FLAGS NOLIBS
    DEPENDENCIES ${boost_library_dependencies} "${BOOST_DATE_TIME_LIBRARY}")

set(delay_baseline_threaded_FLAGS NOLIBS
    DEPENDENCIES
        ${boost_library_dependencies}
        "${BOOST_THREAD_LIBRARY}"
        "${BOOST_DATE_TIME_LIBRARY}")

set(print_heterogeneous_payloads_FLAGS NOLIBS
    DEPENDENCIES ${boost_library_dependencies})

set(benchmarks ${benchmarks}
    boost_tls_overhead
    native_tls_overhead
    hpx_tls_overhead
   )

set(benchmarks ${benchmarks}
    function_object_wrapper_overhead
    coroutines_call_overhead
    serialization_overhead
    future_overhead
    sizeof
   )

set(serialization_overhead_FLAGS DEPENDENCIES iostreams_component)
set(future_overhead_FLAGS DEPENDENCIES iostreams_component)
set(sizeof_FLAGS DEPENDENCIES iostreams_component)

if(HPX_HAVE_CXX11_LAMBDAS)
  set(benchmarks ${benchmarks}
      foreach_scaling
      spinlock_overhead1
      spinlock_overhead2
     )

  set(spinlock_overhead1_FLAGS DEPENDENCIES iostreams_component)
  set(spinlock_overhead2_FLAGS DEPENDENCIES iostreams_component)
  set(foreach_scaling_FLAGS DEPENDENCIES iostreams_component)
endif()


foreach(benchmark ${benchmarks})
  set(sources
      ${benchmark}.cpp)

  source_group("Source Files" FILES ${sources})

  # add example executable
  add_hpx_executable(${benchmark}
                     SOURCES ${sources}
                     ${${benchmark}_FLAGS}
                     FOLDER "Benchmarks/Local/${benchmark}")

  # add a custom target for this example
  add_hpx_pseudo_target(tests.performance.local.${benchmark})

  # make pseudo-targets depend on master pseudo-target
  add_hpx_pseudo_dependencies(tests.performance.local
                              tests.performance.local.${benchmark})

  # add dependencies to pseudo-target
  add_hpx_pseudo_dependencies(tests.performance.local.${benchmark}
                              ${benchmark}_exe)
endforeach()

if(OPENMP_FOUND)
  set_target_properties(openmp_homogeneous_timed_task_spawn_exe PROPERTIES COMPILE_FLAGS ${OpenMP_CXX_FLAGS})
  set_target_properties(openmp_homogeneous_timed_task_spawn_exe PROPERTIES LINK_FLAGS ${OpenMP_CXX_FLAGS})
endif()

add_hpx_pseudo_target(tests.performance.local.htts_v2)
add_subdirectory(htts_v2)
add_hpx_pseudo_dependencies(tests.performance tests.performance.local.htts_v2)

