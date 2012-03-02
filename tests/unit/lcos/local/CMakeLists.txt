# Copyright (c) 2007-2012 Hartmut Kaiser
# Copyright (c) 2011-2012 Bryce Adelstein-Lelbach
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(tests
    local_barrier
    local_mutex_lock
    local_mutex_lock_raii
    local_mutex_lock_sanity
    local_mutex_try_lock
    local_mutex_try_lock_raii
    local_mutex_try_lock_sanity
   )

foreach(test ${tests})
  set(sources
      ${test}.cpp)

  source_group("Source Files" FILES ${sources})

  # add example executable
  add_hpx_executable(${test}_test
                     SOURCES ${sources}
                     ${${test}_FLAGS}
                     FOLDER "Tests/Unit/LCOs/Local")

  # add a custom target for this example
  add_hpx_pseudo_target(tests.unit.lcos.local.${test})

  # make pseudo-targets depend on master pseudo-target
  add_hpx_pseudo_dependencies(tests.unit.lcos.local
                              tests.unit.lcos.local.${test})

  # add dependencies to pseudo-target
  add_hpx_pseudo_dependencies(tests.unit.lcos.local.${test}
                              ${test}_test_exe)
endforeach()

