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
    parse_slurm_nodelist
    stencil3_iterator
    tagged
    transform_iterator
    tuple
   )

if(HPX_WITH_CXX11_STD_INITIALIZER_LIST)
  set(tests ${tests}
    coordinate
  )
endif()

set(subdirs
    bind
    cache
    function
    mem_fn
   )

###############################################################################
if(HWLOC_FOUND)
  set(tests ${tests} parse_affinity_options)
  set(parse_affinity_options_PARAMETERS THREADS_PER_LOCALITY 2)
endif()

set(serialize_buffer_PARAMETERS
    LOCALITIES 2
    THREADS_PER_LOCALITY 2)

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
                     FOLDER "Tests/Unit/Util/")

  add_hpx_unit_test("util" ${test} ${${test}_PARAMETERS})

  # add a custom target for this example
  add_hpx_pseudo_target(tests.unit.util.${test})

  # make pseudo-targets depend on master pseudo-target
  add_hpx_pseudo_dependencies(tests.unit.util
                              tests.unit.util.${test})

  # add dependencies to pseudo-target
  add_hpx_pseudo_dependencies(tests.unit.util.${test}
                              ${test}_test_exe)
endforeach()

# add sub directories
foreach(subdir ${subdirs})
  add_hpx_pseudo_target(tests.unit.util.${subdir}dir)
  add_subdirectory(${subdir})
  add_hpx_pseudo_dependencies(tests.unit.util tests.unit.util.${subdir}dir)
endforeach()
