# Copyright (c) 2011 Bryce Adelstein-Lelbach
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(tests
    any
    any_serialization
    boost_any
    bind_action
    function
    merging_map
    parse_slurm_nodelist
   )

set(merging_map_FLAGS NOLIBS)

if(HWLOC_FOUND)
  set(tests ${tests} parse_affinity_options)
  set(parse_affinity_options_PARAMETERS THREADS_PER_LOCALITY 2)
endif()

foreach(test ${tests})
  set(sources
      ${test}.cpp)

  source_group("Source Files" FILES ${sources})

  # add example executable
  add_hpx_executable(${test}_test
                     SOURCES ${sources}
                     ${${test}_FLAGS}
                     FOLDER "Tests/Unit/Util/")

  add_hpx_unit_test(${test} ${${test}_PARAMETERS})  

  # add a custom target for this example
  add_hpx_pseudo_target(tests.unit.util.${test})

  # make pseudo-targets depend on master pseudo-target
  add_hpx_pseudo_dependencies(tests.unit.util
                              tests.unit.util.${test})

  # add dependencies to pseudo-target
  add_hpx_pseudo_dependencies(tests.unit.util.${test}
                              ${test}_test_exe)
endforeach()

