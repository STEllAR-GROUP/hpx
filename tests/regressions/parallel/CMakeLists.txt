# Copyright (c) 2014-2016 Hartmut Kaiser
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(tests
    for_each_annotated_function
    for_loop_2281
    minimal_findend
    scan_different_inits
    scan_non_commutative
    scan_shortlength
    search_zerolength
    static_chunker_2282
   )

foreach(test ${tests})
  set(sources
      ${test}.cpp)

  set(${test}_PARAMETERS THREADS_PER_LOCALITY 4)

  source_group("Source Files" FILES ${sources})

  # add example executable
  add_hpx_executable(${test}_test
                     SOURCES ${sources}
                     EXCLUDE_FROM_ALL
                     ${${test}_FLAGS}
                     FOLDER "Tests/Regressions/Parallel/")

  add_hpx_regression_test("parallel" ${test} ${${test}_PARAMETERS})

  # add a custom target for this example
  add_hpx_pseudo_target(tests.regressions.parallel_dir.${test})

  # make pseudo-targets depend on master pseudo-target
  add_hpx_pseudo_dependencies(tests.regressions.parallel_dir
                              tests.regressions.parallel_dir.${test})

  # add dependencies to pseudo-target
  add_hpx_pseudo_dependencies(tests.regressions.parallel_dir.${test}
                              ${test}_test_exe)
endforeach()

# add subdirectories
set(subdirs
    executors
   )

if(HPX_WITH_DATAPAR)
  set(subdirs ${subdirs}
      datapar
     )
endif()

foreach(subdir ${subdirs})
  add_hpx_pseudo_target(tests.regressions.parallel_dir.${subdir}_dir)
  add_subdirectory(${subdir})
  add_hpx_pseudo_dependencies(
    tests.regressions.parallel_dir tests.regressions.parallel_dir.${subdir}_dir)
endforeach()

