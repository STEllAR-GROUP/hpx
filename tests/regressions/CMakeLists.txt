# Copyright (c) 2011-2012 Bryce Adelstein-Lelbach
# Copyright (c) 2007-2014 Hartmut Kaiser
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(subdirs
    agas
    actions
    build
    block_matrix
    iostreams
    lcos
    performance_counters
    threads
    traits
    util
    parallel
   )

if(HPX_HAVE_CXX11_LAMBDAS)
    set(subdirs ${subdirs} parallel)
endif()

foreach(subdir ${subdirs})
  add_hpx_pseudo_target(tests.regressions.${subdir}_dir)
  add_subdirectory(${subdir})
  add_hpx_pseudo_dependencies(tests.regressions tests.regressions.${subdir}_dir)
endforeach()

set(tests
    commandline_options_1437
    dynamic_counters_loaded_1508
    id_type_ref_counting_1032
    multiple_init
    unhandled_exception_582
   )

set(commandline_options_1437_PARAMETERS THREADS_PER_LOCALITY 2)
set(id_type_ref_counting_1032_PARAMETERS THREADS_PER_LOCALITY 1)
set(unhandled_exception_582_PARAMETERS THREADS_PER_LOCALITY 1)

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
                     FOLDER "Tests/Regressions/")

  add_hpx_test("tests.regressions" ${test} ${${test}_PARAMETERS})

  # add a custom target for this example
  add_hpx_pseudo_target(tests.regressions.${test})

  # make pseudo-targets depend on master pseudo-target
  add_hpx_pseudo_dependencies(tests.regressions
                              tests.regressions.${test})

  # add dependencies to pseudo-target
  add_hpx_pseudo_dependencies(tests.regressions.${test}
                              ${test}_test_exe)
endforeach()

add_hpx_test(
  "tests.regressions" id_type_ref_counting_1032_4
  EXECUTABLE id_type_ref_counting_1032 THREADS_PER_LOCALITY 4)
