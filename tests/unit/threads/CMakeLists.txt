# Copyright (c) 2011 Bryce Adelstein-Lelbach
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(tests
    lockfree_fifo
    set_thread_state
    thread
    thread_affinity
    thread_id
    thread_launching
    thread_mf
    thread_stacksize
    thread_suspension_executor
    thread_yield
   )

if(HPX_HAVE_THREAD_LOCAL_STORAGE)
  set(tests ${tests} tss)
endif()

if(NOT MSVC)
  set(lockfree_fifo_FLAGS NOLIBS DEPENDENCIES ${Boost_LIBRARIES})
else()
  set(lockfree_fifo_FLAGS NOLIBS)
endif()

set(set_thread_state_PARAMETERS THREADS_PER_LOCALITY 4)

set(thread_affinity_PARAMETERS THREADS_PER_LOCALITY 4)

set(thread_PARAMETERS THREADS_PER_LOCALITY 4)

set(thread_id_PARAMETERS THREADS_PER_LOCALITY 4)

set(thread_launching_PARAMETERS THREADS_PER_LOCALITY 4)

set(thread_mf_PARAMETERS THREADS_PER_LOCALITY 4)

set(thread_stacksize_PARAMETERS LOCALITIES 2)

set(tss_PARAMETERS THREADS_PER_LOCALITY 4)

###############################################################################
foreach(test ${tests})
  set(sources
      ${test}.cpp)

  source_group("Source Files" FILES ${sources})

  # add example executable
  add_hpx_executable(${test}_test
                     SOURCES ${sources}
                     ${${test}_FLAGS}
                     EXCLUDE_FROM_ALL
                     HPX_PREFIX ${HPX_BUILD_PREFIX}
                     FOLDER "Tests/Unit/Threads")

  add_hpx_unit_test("threads" ${test} ${${test}_PARAMETERS})

  # add a custom target for this example
  add_hpx_pseudo_target(tests.unit.threads.${test})

  # make pseudo-targets depend on master pseudo-target
  add_hpx_pseudo_dependencies(tests.unit.threads
                              tests.unit.threads.${test})

  # add dependencies to pseudo-target
  add_hpx_pseudo_dependencies(tests.unit.threads.${test}
                              ${test}_test_exe)
endforeach()

set_property(TARGET lockfree_fifo_test_exe APPEND
    PROPERTY COMPILE_DEFINITIONS
    "HPX_NO_VERSION_CHECK")

