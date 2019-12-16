# Copyright (c) 2019 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(tests
  mpi_ring_async
#  mpi_simple_async
  )

set(mpi_ring_async_PARAMETERS THREADS_PER_LOCALITY 4)
#set(mpi_simple_async_PARAMETERS THREADS_PER_LOCALITY 4)

foreach(test ${tests})

  set(sources
      ${test}.cpp)

  source_group("Source Files" FILES ${sources})

  # add example executable
  add_hpx_executable(${test}_test
    INTERNAL_FLAGS
    SOURCES ${sources}
    ${${test}_FLAGS}
    EXCLUDE_FROM_ALL
    HPX_PREFIX ${HPX_BUILD_PREFIX}
    DEPENDENCIES hpx_mpi hpx_debugging
    FOLDER "Tests/Unit/Modules/MPI")

  add_hpx_unit_test("modules.mpi" ${test} ${${test}_PARAMETERS})

endforeach()

if(HPX_WITH_COMPILE_ONLY_TESTS)
  # add compile time tests
  set(compile_tests
     )

  if(HPX_WITH_FAIL_COMPILE_TESTS)
    set(fail_compile_tests
      )
    foreach(fail_compile_test ${fail_compile_tests})
      set(${fail_compile_test}_FLAGS FAILURE_EXPECTED)
    endforeach()

    set(compile_tests ${compile_tests} ${fail_compile_tests})
  endif()

  foreach(compile_test ${compile_tests})
    set(sources
        ${compile_test}.cpp)

    add_hpx_unit_compile_test(
      "modules.mpi"
      ${compile_test}
      SOURCES ${sources}
      ${${compile_test}_FLAGS}
      FOLDER "Tests/Unit/Modules/MPI/CompileOnly")

  endforeach()

endif()

################################################################################
set(subdirs
   )

# add sub directories
foreach(subdir ${subdirs})
  add_hpx_pseudo_target(tests.unit.modules.mpi.${subdir})
  add_subdirectory(${subdir})
  add_hpx_pseudo_dependencies(tests.unit.modules.mpi
    tests.unit.modules.mpi.${subdir})
endforeach()
