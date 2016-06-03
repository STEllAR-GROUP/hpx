# Copyright (c) 2011-2013 Thomas Heller
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(jacobi_smp_applications)

if(HPX_WITH_CXX11_LAMBDAS)
  set(jacobi_smp_applications ${jacobi_smp_applications}
      jacobi_hpx
      jacobi_nonuniform_hpx)
endif()

if(HPX_WITH_EXAMPLES_OPENMP)
  set(jacobi_smp_applications ${jacobi_smp_applications}
      jacobi_omp_static
      jacobi_omp_dynamic
      jacobi_nonuniform_omp_static
      jacobi_nonuniform_omp_dynamic)
  set(jacobi_omp_static_sources jacobi.cpp)
  set(jacobi_omp_dynamic_sources jacobi.cpp)
  set(jacobi_nonuniform_omp_static_sources jacobi_nonuniform.cpp)
  set(jacobi_nonuniform_omp_dynamic_sources jacobi_nonuniform.cpp)
endif()

set(jacobi_hpx_sources jacobi.cpp)
set(jacobi_nonuniform_hpx_sources jacobi_nonuniform.cpp)

foreach(jacobi_smp_application ${jacobi_smp_applications})
    set(sources
        ${${jacobi_smp_application}_sources} ${jacobi_smp_application}.cpp)

    source_group("Source Files" FILES ${sources})

    # add jacobi_smp executable
    add_hpx_executable(${jacobi_smp_application}
        MODULE jacobi_smp
        SOURCES ${sources}
        ${${jacobi_smp_application}_FLAGS}
        FOLDER "Examples/JacobiSMP")

    # add a custom target for this example
    add_hpx_pseudo_target(examples.jacobi_smp.${jacobi_smp_application})

    # make pseudo-targets depend on master pseudo-target
    add_hpx_pseudo_dependencies(examples.jacobi_smp
        examples.jacobi_smp.${jacobi_smp_application})

    # add dependencies to pseudo-target
    add_hpx_pseudo_dependencies(examples.jacobi_smp.${jacobi_smp_application}
        ${jacobi_smp_application}_exe)
endforeach()

if(HPX_WITH_EXAMPLES_OPENMP)
  set_target_properties(jacobi_omp_static_exe
      PROPERTIES
      COMPILE_DEFINITIONS JACOBI_SMP_NO_HPX)
  set_target_properties(jacobi_omp_static_exe
      PROPERTIES
      COMPILE_FLAGS "${OpenMP_CXX_FLAGS} -DOPENMP_GRID")
  set_target_properties(jacobi_omp_static_exe
      PROPERTIES
      LINK_FLAGS "${OpenMP_CXX_FLAGS}")

  set_target_properties(jacobi_omp_dynamic_exe
      PROPERTIES
      COMPILE_DEFINITIONS JACOBI_SMP_NO_HPX)
  set_target_properties(jacobi_omp_dynamic_exe
      PROPERTIES
      COMPILE_FLAGS "${OpenMP_CXX_FLAGS} -DOPENMP_GRID")
  set_target_properties(jacobi_omp_dynamic_exe
      PROPERTIES
      LINK_FLAGS "${OpenMP_CXX_FLAGS}")

  set_target_properties(jacobi_nonuniform_omp_static_exe
      PROPERTIES
      COMPILE_DEFINITIONS JACOBI_SMP_NO_HPX)
  set_target_properties(jacobi_nonuniform_omp_static_exe
      PROPERTIES
      COMPILE_FLAGS "${OpenMP_CXX_FLAGS} -DOPENMP_GRID")
  set_target_properties(jacobi_nonuniform_omp_static_exe
      PROPERTIES
      LINK_FLAGS "${OpenMP_CXX_FLAGS}")

  set_target_properties(jacobi_nonuniform_omp_dynamic_exe
      PROPERTIES
      COMPILE_DEFINITIONS JACOBI_SMP_NO_HPX)
  set_target_properties(jacobi_nonuniform_omp_dynamic_exe
      PROPERTIES
      COMPILE_FLAGS "${OpenMP_CXX_FLAGS} -DOPENMP_GRID")
  set_target_properties(jacobi_nonuniform_omp_dynamic_exe
      PROPERTIES
      LINK_FLAGS "${OpenMP_CXX_FLAGS}")
endif()
