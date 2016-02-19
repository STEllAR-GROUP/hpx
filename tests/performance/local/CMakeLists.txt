# Copyright (c) 2014-2015 Hartmut Kaiser
# Copyright (c) 2011 Bryce Adelstein-Lelbach
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

################################################################################
# find dependencies
################################################################################
find_package(QThreads)
find_package(TBB)
find_package(SWARM)
find_package(OpenMP)

if(NOT MSVC)
  set(boost_library_dependencies ${Boost_LIBRARIES})
endif()

set(benchmarks
    agas_cache_timings
    async_overheads
    delay_baseline
    delay_baseline_threaded
    hpx_homogeneous_timed_task_spawn_executors
    hpx_heterogeneous_timed_task_spawn
    print_heterogeneous_payloads
    skynet
    timed_task_spawn
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
  include_directories(${QTHREADS_INCLUDE_DIR})

  set(benchmarks
      ${benchmarks}
      qthreads_homogeneous_timed_task_spawn
      qthreads_heterogeneous_timed_task_spawn
     )

  set(qthreads_homogeneous_timed_task_spawn_FLAGS NOLIBS
      DEPENDENCIES ${boost_library_dependencies}
                  ${QTHREADS_LIBRARIES})

  set(qthreads_heterogeneous_timed_task_spawn_FLAGS NOLIBS
      DEPENDENCIES ${boost_library_dependencies}
                  ${QTHREADS_LIBRARIES})
endif()

if(TBB_FOUND)
  include_directories(${TBB_INCLUDE_DIR})

  set(benchmarks
      ${benchmarks}
      tbb_homogeneous_timed_task_spawn
     )

  set(tbb_homogeneous_timed_task_spawn_FLAGS NOLIBS
      DEPENDENCIES ${boost_library_dependencies}
                   ${TBB_LIBRARIES})
endif()

if(SWARM_FOUND)
  include_directories(${SWARM_INCLUDE_DIR})

  set(benchmarks
      ${benchmarks}
      swarm_homogeneous_timed_task_spawn
     )

  set(swarm_homogeneous_timed_task_spawn_FLAGS NOLIBS
      DEPENDENCIES ${boost_library_dependencies}
                   ${SWARM_LIBRARIES})
endif()

set(hpx_homogeneous_timed_task_spawn_executors_FLAGS DEPENDENCIES iostreams_component)
set(hpx_heterogeneous_timed_task_spawn_FLAGS DEPENDENCIES iostreams_component)
set(skynet_FLAGS DEPENDENCIES iostreams_component)

set(delay_baseline_FLAGS NOLIBS
    DEPENDENCIES ${boost_library_dependencies})

set(delay_baseline_threaded_FLAGS NOLIBS
    DEPENDENCIES
        ${boost_library_dependencies})

set(print_heterogeneous_payloads_FLAGS NOLIBS
    DEPENDENCIES ${boost_library_dependencies})

set(benchmarks ${benchmarks}
    boost_tls_overhead
    hpx_tls_overhead
    native_tls_overhead
   )

set(benchmarks ${benchmarks}
    coroutines_call_overhead
    function_object_wrapper_overhead
    future_overhead
    serialization_overhead
    sizeof
   )

set(future_overhead_FLAGS DEPENDENCIES iostreams_component)
set(serialization_overhead_FLAGS DEPENDENCIES iostreams_component)
set(sizeof_FLAGS DEPENDENCIES iostreams_component)

if(HPX_WITH_CXX11_LAMBDAS)
  set(benchmarks ${benchmarks}
      foreach_scaling
      spinlock_overhead1
      spinlock_overhead2
      stencil3_iterators
      stream
      transform_reduce_scaling
      partitioned_vector_foreach
     )

  set(foreach_scaling_FLAGS DEPENDENCIES iostreams_component)
  set(spinlock_overhead1_FLAGS DEPENDENCIES iostreams_component)
  set(spinlock_overhead2_FLAGS DEPENDENCIES iostreams_component)
  set(stencil3_iterators_FLAGS DEPENDENCIES iostreams_component)
  set(stream_FLAGS DEPENDENCIES iostreams_component)
  set(transform_reduce_scaling_FLAGS DEPENDENCIES iostreams_component)
  set(partitioned_vector_foreach_FLAGS
    DEPENDENCIES iostreams_component partitioned_vector_component)
endif()


foreach(benchmark ${benchmarks})
  set(sources
      ${benchmark}.cpp)

  source_group("Source Files" FILES ${sources})

  # add example executable
  add_hpx_executable(${benchmark}
                     SOURCES ${sources}
                     ${${benchmark}_FLAGS}
                     EXCLUDE_FROM_ALL
                     HPX_PREFIX ${HPX_BUILD_PREFIX}
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
  set_target_properties(openmp_homogeneous_timed_task_spawn_exe
    PROPERTIES COMPILE_FLAGS ${OpenMP_CXX_FLAGS})
  set_target_properties(openmp_homogeneous_timed_task_spawn_exe
    PROPERTIES LINK_FLAGS ${OpenMP_CXX_FLAGS})
endif()

add_hpx_pseudo_target(tests.performance.local.htts_v2)
add_subdirectory(htts_v2)
add_hpx_pseudo_dependencies(tests.performance tests.performance.local.htts_v2)

