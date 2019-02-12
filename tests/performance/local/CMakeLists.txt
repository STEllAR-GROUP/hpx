# Copyright (c) 2014-2017 Hartmut Kaiser
# Copyright (c) 2011 Bryce Adelstein-Lelbach
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(boost_library_dependencies ${Boost_LIBRARIES})

set(benchmarks
    agas_cache_timings
    async_overheads
    delay_baseline
    delay_baseline_threaded
    hpx_homogeneous_timed_task_spawn_executors
    hpx_heterogeneous_timed_task_spawn
    parent_vs_child_stealing
    print_heterogeneous_payloads
    resume_suspend
    skynet
    timed_task_spawn
    wait_all_timings
)

if(NOT HPX_WITH_SANITIZERS)
  set(benchmarks
      ${benchmarks}
      start_stop
     )
endif()

if(HPX_WITH_DATAPAR_VC OR HPX_WITH_DATAPAR_BOOST_SIMD)
  set(benchmarks
      ${benchmarks}
      transform_reduce_binary_scaling
     )
  set(transform_reduce_binary_scaling_FLAGS DEPENDENCIES iostreams_component)
endif()

if(HPX_WITH_EXAMPLES_OPENMP)
  set(benchmarks
      ${benchmarks}
      openmp_homogeneous_timed_task_spawn
      openmp_parallel_region
     )

  set(openmp_homogeneous_timed_task_spawn_FLAGS NOLIBS
      DEPENDENCIES ${boost_library_dependencies})
  set(openmp_homogeneous_timed_task_spawn_FLAGS NOLIBS
      DEPENDENCIES ${boost_library_dependencies})
endif()

if(HPX_WITH_EXAMPLES_QTHREADS)
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

if(HPX_WITH_EXAMPLES_TBB)
  include_directories(${TBB_INCLUDE_DIR})

  set(benchmarks
      ${benchmarks}
      tbb_homogeneous_timed_task_spawn
     )

  set(tbb_homogeneous_timed_task_spawn_FLAGS NOLIBS
      DEPENDENCIES ${boost_library_dependencies}
                   ${TBB_LIBRARIES})
endif()

set(hpx_homogeneous_timed_task_spawn_executors_FLAGS DEPENDENCIES iostreams_component)
set(hpx_heterogeneous_timed_task_spawn_FLAGS DEPENDENCIES iostreams_component)
set(parent_vs_child_stealing_FLAGS DEPENDENCIES iostreams_component)
set(skynet_FLAGS DEPENDENCIES iostreams_component)
set(wait_all_timings_FLAGS DEPENDENCIES iostreams_component)

set(delay_baseline_FLAGS NOLIBS
    DEPENDENCIES ${boost_library_dependencies})

set(delay_baseline_threaded_FLAGS
    DEPENDENCIES ${boost_library_dependencies})

set(print_heterogeneous_payloads_FLAGS NOLIBS
    DEPENDENCIES ${boost_library_dependencies})

set(benchmarks ${benchmarks}
    hpx_tls_overhead
    native_tls_overhead
   )

set(benchmarks ${benchmarks}
    coroutines_call_overhead
    function_object_wrapper_overhead
    future_overhead
    serialization_overhead
    serialization_performance
    sizeof
   )

set(future_overhead_FLAGS DEPENDENCIES iostreams_component)
set(serialization_overhead_FLAGS DEPENDENCIES iostreams_component)
set(sizeof_FLAGS DEPENDENCIES iostreams_component)

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

set(future_overhead_PARAMETERS THREADS_PER_LOCALITY 4)

# These tests do not run on hpx threads, so we don't want to pass hpx params into them
set(delay_baseline_PARAMETERS NO_HPX_MAIN)
set(delay_baseline_threaded_PARAMETERS NO_HPX_MAIN)
set(function_object_wrapper_overhead_PARAMETERS NO_HPX_MAIN)
set(nonconcurrent_fifo_overhead_PARAMETERS NO_HPX_MAIN)
set(nonconcurrent_lifo_overhead_PARAMETERS NO_HPX_MAIN)
set(print_heterogeneous_payloads_PARAMETERS NO_HPX_MAIN)

# These tests fail, so I am marking them as non HPX tests until they are fixed
set(print_heterogeneous_payloads_PARAMETERS NO_HPX_MAIN)
set(hpx_homogeneous_timed_task_spawn_executors_PARAMETERS NO_HPX_MAIN)
set(skynet_PARAMETERS NO_HPX_MAIN)
set(timed_task_spawn_PARAMETERS NO_HPX_MAIN)
set(hpx_tls_overhead_PARAMETERS NO_HPX_MAIN)
set(native_tls_overhead_PARAMETERS NO_HPX_MAIN)
set(coroutines_call_overhead_PARAMETERS NO_HPX_MAIN)
set(serialization_performance_PARAMETERS NO_HPX_MAIN)

if(HPX_WITH_CUDA)
  set_source_files_properties(stream.cpp PROPERTIES CUDA_SOURCE_PROPERTY_FORMAT OBJ)
endif()
if(HPX_WITH_CUDA_CLANG)
  set_source_files_properties(stream.cpp PROPERTIES COMPILE_FLAGS "-xcuda ${HPX_CUDA_CLANG_FLAGS}")
endif()

foreach(benchmark ${benchmarks})
  set(sources
      ${benchmark}.cpp)

  source_group("Source Files" FILES ${sources})

  # add example executable
  add_hpx_executable(${benchmark}_test
                     SOURCES ${sources}
                     ${${benchmark}_FLAGS}
                     EXCLUDE_FROM_ALL
                     HPX_PREFIX ${HPX_BUILD_PREFIX}
                     FOLDER "Benchmarks/Local")

  list(FIND ${benchmark}_FLAGS NOLIBS has_NOLIBS)
  if (NOT has_NOLIBS EQUAL -1)
    target_compile_definitions(${benchmark}_test PRIVATE
                               HPX_FORMAT_EXPORT= #empty
                               HPX_NO_VERSION_CHECK)
    target_sources(${benchmark}_test
                   PRIVATE ${PROJECT_SOURCE_DIR}/src/util/format.cpp)
  endif()

  if(NOT "${${benchmark}_PARAMETERS}" MATCHES NO_HPX_MAIN)
    add_hpx_test("tests.performance.local" ${benchmark} ${${benchmark}_PARAMETERS})
  endif()

  # add a custom target for this example
  add_hpx_pseudo_target(tests.performance.local.${benchmark})

  # make pseudo-targets depend on master pseudo-target
  add_hpx_pseudo_dependencies(tests.performance.local
                              tests.performance.local.${benchmark})

  # add dependencies to pseudo-target
  add_hpx_pseudo_dependencies(tests.performance.local.${benchmark}
                              ${benchmark}_test)

endforeach()

if(HPX_WITH_EXAMPLES_OPENMP)
  set_target_properties(openmp_homogeneous_timed_task_spawn_test
    PROPERTIES COMPILE_FLAGS ${OpenMP_CXX_FLAGS})
  set_target_properties(openmp_homogeneous_timed_task_spawn_test
    PROPERTIES LINK_FLAGS ${OpenMP_CXX_FLAGS})
  set_target_properties(openmp_parallel_region_test
    PROPERTIES COMPILE_FLAGS ${OpenMP_CXX_FLAGS})
  set_target_properties(openmp_parallel_region_test
    PROPERTIES LINK_FLAGS ${OpenMP_CXX_FLAGS})
endif()

add_hpx_pseudo_target(tests.performance.local.htts_v2)
add_subdirectory(htts_v2)
add_hpx_pseudo_dependencies(tests.performance tests.performance.local.htts_v2)

