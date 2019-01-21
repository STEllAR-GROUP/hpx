# Copyright (c) 2011-2014 Bryce Adelstein-Lelbach
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

if((NOT MSVC) OR HPX_WITH_VCPKG)
  set(boost_library_dependencies ${Boost_LIBRARIES})
endif()

set(benchmarks
    htts2_payload_precision
#    htts2_payload_baseline
    htts2_hpx
   )

set(htts2_payload_precision_FLAGS NOLIBS DEPENDENCIES ${boost_library_dependencies} hpx.config)

if(HPX_WITH_EXAMPLES_OPENMP)
  set(benchmarks ${benchmarks} htts2_omp)
  set(htts2_omp_FLAGS NOLIBS DEPENDENCIES ${boost_library_dependencies} hpx.config)
endif()

if(HPX_WITH_EXAMPLES_QTHREADS)
  include_directories(${QTHREADS_INCLUDE_DIR})

  set(benchmarks ${benchmarks} htts2_qthreads)
  set(htts2_qthreads_FLAGS NOLIBS DEPENDENCIES ${boost_library_dependencies} ${QTHREADS_LIBRARY} hpx.config)
endif()

if(HPX_WITH_EXAMPLES_TBB)
  include_directories(${TBB_INCLUDE_DIR})

  set(benchmarks ${benchmarks} htts2_tbb)
  set(htts2_tbb_FLAGS NOLIBS DEPENDENCIES ${boost_library_dependencies} ${TBB_LIBRARY} hpx.config)
endif()

foreach(benchmark ${benchmarks})
  set(sources htts2.cpp ${benchmark}.cpp)

  source_group("Source Files" FILES ${sources})

  # add example executable
  add_hpx_executable(${benchmark}
                     SOURCES ${sources}
                     ${${benchmark}_FLAGS}
                     EXCLUDE_FROM_ALL
                     HPX_PREFIX ${HPX_BUILD_PREFIX}
                     FOLDER "Benchmarks/HTTS v2/${benchmark}")

  # add a custom target for this example
  add_hpx_pseudo_target(tests.performance.local.htts_v2.${benchmark})

  # make pseudo-targets depend on master pseudo-target
  add_hpx_pseudo_dependencies(tests.performance.local.htts_v2
                              tests.performance.local.htts_v2.${benchmark})

  # add dependencies to pseudo-target
  add_hpx_pseudo_dependencies(tests.performance.local.htts_v2.${benchmark}
                              ${benchmark})
endforeach()

target_include_directories(htts2_payload_precision PRIVATE ${HPX_SOURCE_DIR})
if(HPX_WITH_EXAMPLES_OPENMP)
  set_target_properties(htts2_omp PROPERTIES COMPILE_FLAGS ${OpenMP_CXX_FLAGS})
  set_target_properties(htts2_omp PROPERTIES LINK_FLAGS ${OpenMP_CXX_FLAGS})
  target_include_directories(htts2_omp PRIVATE ${HPX_SOURCE_DIR})
endif()
if(HPX_WITH_EXAMPLES_OPENMP)
  target_include_directories(htts2_tbb PRIVATE ${HPX_SOURCE_DIR})
endif()
if(HPX_WITH_EXAMPLES_QTHREADS)
  target_include_directories(htts2_qthreads PRIVATE ${HPX_SOURCE_DIR})
endif()

