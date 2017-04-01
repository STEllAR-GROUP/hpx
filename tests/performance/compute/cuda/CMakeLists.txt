# Copyright (c) 2014-2015 Hartmut Kaiser
# Copyright (c) 2011 Bryce Adelstein-Lelbach
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(benchmarks
  synchronize)

if(HPX_WITH_CUDA)
  set_source_files_properties(synchronize.cpp PROPERTIES CUDA_SOURCE_PROPERTY_FORMAT OBJ)
endif()

foreach(benchmark ${benchmarks})
  set(sources
      ${benchmark}.cpp)

  source_group("Source Files" FILES ${sources})

  # add example executable
  add_hpx_executable(${benchmark}
                     SOURCES ${sources}
                     ${${benchmark}_FLAGS}
                     EXCLUDE_FROM_ALL
                     HPX_PREFIX ${HPX_BUILD_PREFIX}
                     FOLDER "Benchmarks/Compute/Cuda")

  # add a custom target for this example
  add_hpx_pseudo_target(tests.performance.compute.cuda.${benchmark})

  # make pseudo-targets depend on master pseudo-target
  add_hpx_pseudo_dependencies(tests.performance.compute.cuda
                              tests.performance.compute.cuda.${benchmark})

  # add dependencies to pseudo-target
  add_hpx_pseudo_dependencies(tests.performance.compute.cuda.${benchmark}
                              ${benchmark}_exe)
endforeach()
