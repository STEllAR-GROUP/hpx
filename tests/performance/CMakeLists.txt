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

set(benchmarks
    boost_tls_overhead
    coroutines_call_overhead
    delay_baseline
    function_object_wrapper_overhead
    hpx_tls_overhead
    hpx_homogeneous_task_spawn
    hpx_homogeneous_task_spawn_executors
    hpx_homogeneous_timed_task_spawn
    hpx_homogeneous_timed_task_spawn_executors
    hpx_heterogeneous_task_spawn
    native_tls_overhead
    print_heterogeneous_payloads
    sizeof
   )

add_hpx_library(test_worker
    SOURCES worker.cpp
    HEADERS worker.hpp
    FOLDER "Benchmarks/Worker")
    
add_hpx_library(test_worker_timed
    SOURCES worker_timed.cpp
    HEADERS worker_timed.hpp
    FOLDER "Benchmarks/Worker_Timed")

if(MSVC)
  set(boost_library_dependencies)
else()
    set(boost_library_dependencies ${BOOST_PROGRAM_OPTIONS_LIBRARY} ${BOOST_SYSTEM_LIBRARY} ${BOOST_CHRONO_LIBRARY})
endif()

if(OPENMP_FOUND)
  set(benchmarks
      ${benchmarks}
      openmp_homogeneous_task_spawn
     )

  set(openmp_homogeneous_task_spawn_FLAGS NOLIBS
      DEPENDENCIES ${boost_library_dependencies} test_worker_lib)
endif()

if(UNIX)
  # GCC 4.7 chokes on Qthreads headers.
  if(QTHREADS_FOUND AND (040700 GREATER ${GCC_VERSION}))
    include_directories(${QTHREADS_INCLUDE_DIR})

    set(benchmarks
        ${benchmarks}
        qthreads_homogeneous_task_spawn
        qthreads_homogeneous_timed_task_spawn
        qthreads_heterogeneous_task_spawn
       )

    set(qthreads_homogeneous_task_spawn_FLAGS NOLIBS
        DEPENDENCIES ${boost_library_dependencies}
                     ${QTHREADS_LIBRARY} test_worker_lib)

    set(qthreads_homogeneous_timed_task_spawn_FLAGS NOLIBS
        DEPENDENCIES ${boost_library_dependencies}
                     ${QTHREADS_LIBRARY} test_worker_timed_lib)

    set(qthreads_heterogeneous_task_spawn_FLAGS NOLIBS
        DEPENDENCIES ${boost_library_dependencies}
                     ${QTHREADS_LIBRARY} test_worker_lib)
  endif()
endif()

if(TBB_FOUND)
  include_directories(${TBB_INCLUDE_DIR})

  set(benchmarks
      ${benchmarks}
      tbb_homogeneous_task_spawn
     )

  set(tbb_homogeneous_task_spawn_FLAGS NOLIBS
      DEPENDENCIES ${boost_library_dependencies}
                   ${TBB_LIBRARY} test_worker_lib)
endif()

if(SWARM_FOUND)
  include_directories(${SWARM_INCLUDE_DIR})

  set(benchmarks
      ${benchmarks}
      swarm_homogeneous_task_spawn
     )

  set(swarm_homogeneous_task_spawn_FLAGS NOLIBS
      DEPENDENCIES ${boost_library_dependencies}
                   ${SWARM_LIBRARY} test_worker_lib)
endif()

set(delay_baseline_FLAGS NOLIBS
    DEPENDENCIES ${boost_library_dependencies} test_worker_lib)

set(print_heterogeneous_payloads_FLAGS NOLIBS
    DEPENDENCIES ${boost_library_dependencies})

if(NOT MSVC)
  set(boost_library_dependencies ${boost_library_dependencies} ${BOOST_THREAD_LIBRARY})
endif()

set(boost_tls_overhead_FLAGS NOLIBS
    DEPENDENCIES ${boost_library_dependencies})

set(native_tls_overhead_FLAGS NOLIBS
    DEPENDENCIES ${boost_library_dependencies})

set(hpx_tls_overhead_FLAGS NOLIBS
    DEPENDENCIES ${boost_library_dependencies})

set(hpx_homogeneous_task_spawn_FLAGS DEPENDENCIES iostreams_component test_worker_lib)
set(hpx_homogeneous_task_spawn_executors_FLAGS DEPENDENCIES iostreams_component test_worker_lib)
set(hpx_homogeneous_timed_task_spawn_FLAGS DEPENDENCIES iostreams_component test_worker_timed_lib)
set(hpx_homogeneous_timed_task_spawn_executors_FLAGS DEPENDENCIES iostreams_component test_worker_timed_lib)

set(hpx_heterogeneous_task_spawn_FLAGS DEPENDENCIES iostreams_component test_worker_lib)

if(HPX_HAVE_CXX11_LAMBDAS)
  set(benchmarks ${benchmarks}
      future_overhead
      spinlock_overhead1
      spinlock_overhead2
     )

  set(future_overhead_FLAGS DEPENDENCIES iostreams_component)

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

