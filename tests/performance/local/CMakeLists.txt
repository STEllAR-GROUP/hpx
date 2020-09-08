# Copyright (c) 2014-2017 Hartmut Kaiser
# Copyright (c) 2011 Bryce Adelstein-Lelbach
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(boost_library_dependencies ${Boost_LIBRARIES})

set(benchmarks
    async_overheads
    coroutines_call_overhead
    delay_baseline
    delay_baseline_threaded
    function_object_wrapper_overhead
    future_overhead
    hpx_tls_overhead
    native_tls_overhead
    print_heterogeneous_payloads
    resume_suspend
    timed_task_spawn
)

if(NOT HPX_WITH_CUDA)
  list(APPEND benchmarks stream)
endif()

if(NOT HPX_WITH_SANITIZERS)
  list(APPEND benchmarks start_stop)
  set(start_stop_FLAGS DEPENDENCIES hpx_timing)
endif()

if(HPX_WITH_LIBCDS)
  list(APPEND benchmarks libcds_hazard_pointer_overhead)
  set(libcds_hazard_pointer_overhead_FLAGS DEPENDENCIES iostreams_component)
endif()

if(HPX_WITH_DISTRIBUTED_RUNTIME AND (HPX_WITH_DATAPAR_VC
                                     OR HPX_WITH_DATAPAR_BOOST_SIMD)
)
  list(APPEND benchmarks transform_reduce_binary_scaling)
  set(transform_reduce_binary_scaling_FLAGS DEPENDENCIES iostreams_component
                                            hpx_timing
  )
endif()

if(HPX_WITH_EXAMPLES_OPENMP)
  list(APPEND benchmarks openmp_homogeneous_timed_task_spawn
       openmp_parallel_region
  )

  set(openmp_homogeneous_timed_task_spawn_FLAGS
      NOLIBS
      DEPENDENCIES
      ${boost_library_dependencies}
      hpx_assertion
      hpx_config
      hpx_format
      hpx_timing
  )
  set(openmp_parallel_region_FLAGS
      NOLIBS DEPENDENCIES ${boost_library_dependencies} hpx_format hpx_timing
  )
endif()

if(HPX_WITH_EXAMPLES_QTHREADS)
  list(APPEND benchmarks qthreads_homogeneous_timed_task_spawn
       qthreads_heterogeneous_timed_task_spawn
  )

  set(qthreads_homogeneous_timed_task_spawn_FLAGS
      NOLIBS
      DEPENDENCIES
      ${boost_library_dependencies}
      ${QTHREADS_LIBRARIES}
      hpx_config
      hpx_format
      hpx_preprocessor
      hpx_timing
  )

  set(qthreads_heterogeneous_timed_task_spawn_FLAGS
      NOLIBS
      DEPENDENCIES
      ${boost_library_dependencies}
      ${QTHREADS_LIBRARIES}
      hpx_assertion
      hpx_format
      hpx_timing
  )

  set(qthreads_heterogeneous_timed_task_spawn_INCLUDE_DIRECTORIES
      ${QTHREADS_INCLUDE_DIR}
  )
endif()

if(HPX_WITH_EXAMPLES_TBB)
  list(APPEND benchmarks tbb_homogeneous_timed_task_spawn)

  set(tbb_homogeneous_timed_task_spawn_FLAGS
      NOLIBS
      DEPENDENCIES
      ${boost_library_dependencies}
      ${TBB_LIBRARIES}
      hpx_config
      hpx_format
      hpx_preprocessor
      hpx_timing
  )

  set(tbb_homogeneous_timed_task_spawn_INCLUDE_DIRECTORIES ${TBB_INCLUDE_DIR})
endif()

if(HPX_WITH_DISTRIBUTED_RUNTIME)
  list(
    APPEND
    benchmarks
    agas_cache_timings
    foreach_scaling
    hpx_homogeneous_timed_task_spawn_executors
    hpx_heterogeneous_timed_task_spawn
    parent_vs_child_stealing
    partitioned_vector_foreach
    skynet
    sizeof
    spinlock_overhead1
    spinlock_overhead2
    wait_all_timings
  )
endif()

set(delay_baseline_FLAGS NOLIBS DEPENDENCIES ${boost_library_dependencies}
                         hpx_config hpx_format hpx_timing
)

set(delay_baseline_threaded_FLAGS DEPENDENCIES ${boost_library_dependencies}
                                  hpx_format hpx_timing
)

set(print_heterogeneous_payloads_FLAGS
    NOLIBS DEPENDENCIES ${boost_library_dependencies} hpx_config hpx_format
)
set(resume_suspend_FLAGS DEPENDENCIES hpx_timing)

set(native_tls_overhead_LIBRARIES hpx_dependencies_boost)

set(hpx_homogeneous_timed_task_spawn_executors_FLAGS DEPENDENCIES
                                                     iostreams_component
)
set(hpx_heterogeneous_timed_task_spawn_FLAGS DEPENDENCIES iostreams_component
                                             hpx_timing
)
set(parent_vs_child_stealing_FLAGS DEPENDENCIES iostreams_component hpx_timing)
set(skynet_FLAGS DEPENDENCIES iostreams_component)
set(wait_all_timings_FLAGS DEPENDENCIES iostreams_component hpx_timing)
set(future_overhead_FLAGS DEPENDENCIES hpx_timing)
set(sizeof_FLAGS DEPENDENCIES iostreams_component)
set(foreach_scaling_FLAGS DEPENDENCIES iostreams_component hpx_timing)
set(spinlock_overhead1_FLAGS DEPENDENCIES iostreams_component hpx_timing)
set(spinlock_overhead2_FLAGS DEPENDENCIES iostreams_component hpx_timing)
set(partitioned_vector_foreach_FLAGS DEPENDENCIES iostreams_component
                                     partitioned_vector_component hpx_timing
)

set(function_object_wrapper_overhead_FLAGS DEPENDENCIES hpx_timing)
set(hpx_tls_overhead_FLAGS DEPENDENCIES hpx_timing)
set(native_tls_overhead_FLAGS DEPENDENCIES hpx_timing)
set(nonconcurrent_fifo_overhead_FLAGS DEPENDENCIES hpx_timing)
set(nonconcurrent_lifo_overhead_FLAGS DEPENDENCIES hpx_timing)
set(transform_reduce_scaling_FLAGS DEPENDENCIES hpx_timing)
set(future_overhead_PARAMETERS THREADS_PER_LOCALITY 4)

# These tests do not run on hpx threads, so we don't want to pass hpx params
# into them
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

if(HPX_WITH_CUDA_COMPUTE)
  set_source_files_properties(
    stream.cpp PROPERTIES CUDA_SOURCE_PROPERTY_FORMAT OBJ
  )
endif()
if(HPX_WITH_CUDA_CLANG)
  set_source_files_properties(
    stream.cpp PROPERTIES COMPILE_FLAGS "-xcuda ${HPX_CUDA_CLANG_FLAGS}"
  )
endif()

foreach(benchmark ${benchmarks})
  set(sources ${benchmark}.cpp)

  source_group("Source Files" FILES ${sources})

  # add example executable
  add_hpx_executable(
    ${benchmark}_test INTERNAL_FLAGS
    SOURCES ${sources} ${${benchmark}_FLAGS}
    EXCLUDE_FROM_ALL
    HPX_PREFIX ${HPX_BUILD_PREFIX}
    FOLDER "Benchmarks/Local"
  )

  target_include_directories(
    ${benchmark}_test SYSTEM PRIVATE ${${benchmark}_INCLUDE_DIRECTORIES}
  )
  target_link_libraries(${benchmark}_test PRIVATE ${${benchmark}_LIBRARIES})

  if(NOT "${${benchmark}_PARAMETERS}" MATCHES NO_HPX_MAIN)
    add_hpx_performance_test("local" ${benchmark} ${${benchmark}_PARAMETERS})
  endif()

endforeach()

if(HPX_WITH_LIBCDS)
  target_link_libraries(libcds_hazard_pointer_overhead_test PRIVATE cds)
endif()

if(HPX_WITH_EXAMPLES_OPENMP)
  set_target_properties(
    openmp_homogeneous_timed_task_spawn_test PROPERTIES COMPILE_FLAGS
                                                        ${OpenMP_CXX_FLAGS}
  )
  set_target_properties(
    openmp_homogeneous_timed_task_spawn_test PROPERTIES LINK_FLAGS
                                                        ${OpenMP_CXX_FLAGS}
  )
  set_target_properties(
    openmp_parallel_region_test PROPERTIES COMPILE_FLAGS ${OpenMP_CXX_FLAGS}
  )
  set_target_properties(
    openmp_parallel_region_test PROPERTIES LINK_FLAGS ${OpenMP_CXX_FLAGS}
  )
endif()

add_hpx_pseudo_target(tests.performance.local.htts_v2)
add_subdirectory(htts_v2)
add_hpx_pseudo_dependencies(
  tests.performance.local tests.performance.local.htts_v2
)
