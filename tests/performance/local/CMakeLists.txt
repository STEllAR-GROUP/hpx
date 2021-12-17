# Copyright (c) 2014-2017 Hartmut Kaiser
# Copyright (c) 2011 Bryce Adelstein-Lelbach
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(benchmarks)

if(NOT HPXLocal_WITH_CUDA)
  list(APPEND benchmarks stream stream_report)
  set(stream_FLAGS CUDA)
endif()

if(HPX_WITH_LIBCDS)
  list(APPEND benchmarks libcds_hazard_pointer_overhead)
  set(libcds_hazard_pointer_overhead_FLAGS DEPENDENCIES iostreams_component)
endif()

list(
  APPEND
  benchmarks
  agas_cache_timings
  future_overhead
  future_overhead_report
  hpx_homogeneous_timed_task_spawn_executors
  partitioned_vector_foreach
  sizeof
  spinlock_overhead1
  spinlock_overhead2
)

set(sizeof_FLAGS DEPENDENCIES iostreams_component)
set(spinlock_overhead1_FLAGS DEPENDENCIES iostreams_component)
set(spinlock_overhead2_FLAGS DEPENDENCIES iostreams_component)
set(partitioned_vector_foreach_FLAGS DEPENDENCIES iostreams_component
                                     partitioned_vector_component
)

set(future_overhead_PARAMETERS THREADS_PER_LOCALITY 4)
set(future_overhead_report_PARAMETERS THREADS_PER_LOCALITY 4)

# These tests fail, so I am marking them as non HPX tests until they are fixed
set(hpx_homogeneous_timed_task_spawn_executors_PARAMETERS NO_HPX_MAIN)

foreach(benchmark ${benchmarks})
  set(sources ${benchmark}.cpp ${${benchmark}_SOURCES})

  source_group("Source Files" FILES ${sources})
  if(${benchmark}_HEADERS)
    source_group("Header Files" FILES ${${benchmark}_HEADERS})
    set(${benchmark}_HEADERS HEADERS ${${benchmark}_HEADERS})
  endif()

  # add example executable
  add_hpx_executable(
    ${benchmark}_test INTERNAL_FLAGS
    SOURCES ${sources} ${${benchmark}_FLAGS} ${${benchmark}_HEADERS}
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
