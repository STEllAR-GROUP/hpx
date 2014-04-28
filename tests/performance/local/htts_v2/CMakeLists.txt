# Copyright (c) 2011-2014 Bryce Adelstein-Lelbach
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

################################################################################
# Find dependencies
################################################################################
#find_package(HPX_Qthreads)
#find_package(HPX_TBB)
#find_package(HPX_SWARM)
#find_package(HPX_OpenMP)

if(MSVC)
  set(boost_library_dependencies)
else()
  set(boost_library_dependencies
      ${BOOST_PROGRAM_OPTIONS_LIBRARY}
      ${BOOST_SYSTEM_LIBRARY}
      ${BOOST_DATE_TIME_LIBRARY}
      ${BOOST_CHRONO_LIBRARY})
endif()

set(benchmarks
    htts2_payload_precision
#    htts2_payload_baseline
    htts2_hpx
   )

set(htts2_payload_precision_FLAGS NOLIBS DEPENDENCIES ${boost_library_dependencies})

if(OPENMP_FOUND)
  set(benchmarks ${benchmarks} htts2_omp)
  set(htts2_omp_FLAGS NOLIBS DEPENDENCIES ${boost_library_dependencies})
endif()

if(QTHREADS_FOUND)
  include_directories(${QTHREADS_INCLUDE_DIR})

  set(benchmarks ${benchmarks} htts2_qthreads)
  set(htts2_qthreads_FLAGS NOLIBS DEPENDENCIES ${boost_library_dependencies} ${QTHREADS_LIBRARY})
endif()

if(TBB_FOUND)
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
                     FOLDER "Benchmarks/HTTS v2/${benchmark}")

  # add a custom target for this example
  add_hpx_pseudo_target(tests.performance.local.htts_v2.${benchmark})

  # make pseudo-targets depend on master pseudo-target
  add_hpx_pseudo_dependencies(tests.performance.local.htts_v2
                              tests.performance.local.htts_v2.${benchmark})

  # add dependencies to pseudo-target
  add_hpx_pseudo_dependencies(tests.performance.local.htts_v2.${benchmark}
                              ${benchmark}_exe)
endforeach()

if(OPENMP_FOUND)
  set_target_properties(htts2_omp_exe PROPERTIES COMPILE_FLAGS ${OpenMP_CXX_FLAGS})
  set_target_properties(htts2_omp_exe PROPERTIES LINK_FLAGS ${OpenMP_CXX_FLAGS})
endif()

