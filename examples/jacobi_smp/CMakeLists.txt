# Copyright (c) 2011-2013 Thomas Heller
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(jacobi_smp_applications jacobi_hpx jacobi_nonuniform_hpx)

set(jacobi_hpx_PARAMETERS THREADS_PER_LOCALITY 4)
set(jacobi_nonuniform_hpx_PARAMETERS THREADS_PER_LOCALITY 4)

set(disabled_tests
    # Disabled because requires external input data. TODO: Download data when
    # running test.
    jacobi_nonuniform_hpx
)

if(HPX_WITH_EXAMPLES_OPENMP)
  set(jacobi_smp_applications
      ${jacobi_smp_applications} jacobi_omp_static jacobi_omp_dynamic
      jacobi_nonuniform_omp_static jacobi_nonuniform_omp_dynamic
  )
  set(jacobi_omp_static_sources jacobi.cpp)
  set(jacobi_omp_dynamic_sources jacobi.cpp)
  set(jacobi_nonuniform_omp_static_sources jacobi_nonuniform.cpp)
  set(jacobi_nonuniform_omp_dynamic_sources jacobi_nonuniform.cpp)

  set(disabled_tests ${disabled_tests} jacobi_nonuniform_omp_static
                     jacobi_nonuniform_omp_dynamic
  )
endif()

set(jacobi_hpx_sources jacobi.cpp)
set(jacobi_nonuniform_hpx_sources jacobi_nonuniform.cpp)

foreach(jacobi_smp_application ${jacobi_smp_applications})
  set(sources ${${jacobi_smp_application}_sources}
              ${jacobi_smp_application}.cpp
  )

  source_group("Source Files" FILES ${sources})

  # add jacobi_smp executable
  add_hpx_executable(
    ${jacobi_smp_application} INTERNAL_FLAGS MODULE jacobi_smp
    SOURCES ${sources} ${${jacobi_smp_application}_FLAGS}
    FOLDER "Examples/JacobiSMP"
  )

  add_hpx_example_target_dependencies("jacobi_smp" ${jacobi_smp_application})

  if(HPX_WITH_TESTS
     AND HPX_WITH_TESTS_EXAMPLES
     AND NOT ("${jacobi_smp_application}" IN_LIST disabled_tests)
  )
    add_hpx_example_test(
      "jacobi_smp" ${jacobi_smp_application}
      ${${jacobi_smp_application}_PARAMETERS}
    )
  endif()
endforeach()

if(HPX_WITH_EXAMPLES_OPENMP)
  set_target_properties(
    jacobi_omp_static PROPERTIES COMPILE_DEFINITIONS JACOBI_SMP_NO_HPX
  )
  set_target_properties(
    jacobi_omp_static PROPERTIES COMPILE_FLAGS
                                 "${OpenMP_CXX_FLAGS} -DOPENMP_GRID"
  )
  set_target_properties(
    jacobi_omp_static PROPERTIES LINK_FLAGS "${OpenMP_CXX_FLAGS}"
  )

  set_target_properties(
    jacobi_omp_dynamic PROPERTIES COMPILE_DEFINITIONS JACOBI_SMP_NO_HPX
  )
  set_target_properties(
    jacobi_omp_dynamic PROPERTIES COMPILE_FLAGS
                                  "${OpenMP_CXX_FLAGS} -DOPENMP_GRID"
  )
  set_target_properties(
    jacobi_omp_dynamic PROPERTIES LINK_FLAGS "${OpenMP_CXX_FLAGS}"
  )

  set_target_properties(
    jacobi_nonuniform_omp_static PROPERTIES COMPILE_DEFINITIONS
                                            JACOBI_SMP_NO_HPX
  )
  set_target_properties(
    jacobi_nonuniform_omp_static PROPERTIES COMPILE_FLAGS
                                            "${OpenMP_CXX_FLAGS} -DOPENMP_GRID"
  )
  set_target_properties(
    jacobi_nonuniform_omp_static PROPERTIES LINK_FLAGS "${OpenMP_CXX_FLAGS}"
  )

  set_target_properties(
    jacobi_nonuniform_omp_dynamic PROPERTIES COMPILE_DEFINITIONS
                                             JACOBI_SMP_NO_HPX
  )
  set_target_properties(
    jacobi_nonuniform_omp_dynamic
    PROPERTIES COMPILE_FLAGS "${OpenMP_CXX_FLAGS} -DOPENMP_GRID"
  )
  set_target_properties(
    jacobi_nonuniform_omp_dynamic PROPERTIES LINK_FLAGS "${OpenMP_CXX_FLAGS}"
  )
endif()
