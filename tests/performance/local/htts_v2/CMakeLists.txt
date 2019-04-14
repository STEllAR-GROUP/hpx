# Copyright (c) 2011-2014 Bryce Adelstein-Lelbach
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(boost_library_dependencies ${Boost_LIBRARIES})

set(benchmarks
    htts2_payload_precision
#    htts2_payload_baseline
    htts2_hpx
   )

set(htts2_payload_precision_FLAGS NOLIBS DEPENDENCIES ${boost_library_dependencies})

if(HPX_WITH_EXAMPLES_OPENMP)
  set(benchmarks ${benchmarks} htts2_omp)
  set(htts2_omp_FLAGS NOLIBS DEPENDENCIES ${boost_library_dependencies})
endif()

if(HPX_WITH_EXAMPLES_QTHREADS)
  include_directories(${QTHREADS_INCLUDE_DIR})

  set(benchmarks ${benchmarks} htts2_qthreads)
  set(htts2_qthreads_FLAGS NOLIBS DEPENDENCIES ${boost_library_dependencies} ${QTHREADS_LIBRARY})
endif()

if(HPX_WITH_EXAMPLES_TBB)
  include_directories(${TBB_INCLUDE_DIR})

  set(benchmarks ${benchmarks} htts2_tbb)
  set(htts2_tbb_FLAGS NOLIBS DEPENDENCIES ${boost_library_dependencies} ${TBB_LIBRARY})
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

  list(FIND ${benchmark}_FLAGS NOLIBS has_NOLIBS)
  if (NOT has_NOLIBS EQUAL -1)
    target_compile_definitions(${benchmark} PRIVATE
                               HPX_FORMAT_EXPORT= #empty
                               HPX_NO_VERSION_CHECK)
    target_sources(${benchmark}
                   PRIVATE ${PROJECT_SOURCE_DIR}/src/util/format.cpp)
  endif()

  # add a custom target for this example
  add_hpx_pseudo_target(tests.performance.local.htts_v2.${benchmark})

  # make pseudo-targets depend on master pseudo-target
  add_hpx_pseudo_dependencies(tests.performance.local.htts_v2
                              tests.performance.local.htts_v2.${benchmark})

  # add dependencies to pseudo-target
  add_hpx_pseudo_dependencies(tests.performance.local.htts_v2.${benchmark}
                              ${benchmark})
endforeach()

if(HPX_WITH_EXAMPLES_OPENMP)
  set_target_properties(htts2_omp PROPERTIES COMPILE_FLAGS ${OpenMP_CXX_FLAGS})
  set_target_properties(htts2_omp PROPERTIES LINK_FLAGS ${OpenMP_CXX_FLAGS})
endif()

