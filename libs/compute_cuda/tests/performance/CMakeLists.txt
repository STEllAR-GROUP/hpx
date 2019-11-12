# Copyright (c) 2014-2015 Hartmut Kaiser
# Copyright (c) 2011 Bryce Adelstein-Lelbach
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(benchmarks
  synchronize)

if(HPX_WITH_CUDA)
  set_source_files_properties(synchronize.cpp PROPERTIES CUDA_SOURCE_PROPERTY_FORMAT OBJ)
endif()

if(HPX_WITH_CUDA_CLANG)
  set_source_files_properties(synchronize.cpp PROPERTIES COMPILE_FLAGS -xcuda)
endif()

foreach(benchmark ${benchmarks})
  set(sources
      ${benchmark}.cpp)

  source_group("Source Files" FILES ${sources})

  # add example executable
  add_hpx_executable(${benchmark}_test
    INTERNAL_FLAGS
    SOURCES ${sources}
    ${${benchmark}_FLAGS}
    EXCLUDE_FROM_ALL
    HPX_PREFIX ${HPX_BUILD_PREFIX}
    FOLDER "Benchmarks/Modules/Compute/Cuda")

  add_hpx_performance_test("modules.compute_cuda" ${benchmark} ${${benchmark}_PARAMETERS})
endforeach()
