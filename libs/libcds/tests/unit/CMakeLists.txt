# Copyright (c) 2019 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(tests libcds_map_hazard_pointer_hpxthread
          libcds_map_hazard_pointer_stdthread
)

# Flat combining test uses boost thread for thread_specific_ptr
if(NOT TARGET Boost::thread)
  find_package(
    Boost ${Boost_MINIMUM_VERSION} MODULE
    COMPONENTS thread
    QUIET
  )
endif()

if(Boost_THREAD_FOUND)
  set(tests ${tests} libcds_flat_combining_priority_queue_stdthread)
  set(libcds_flat_combining_priority_queue_stdthread THREADS_PER_LOCALITY 4)
endif()

# make sure we set the concurrent thread limit high enough
set(libcds_map_hazard_pointer_hpxthread_PARAMETERS
    THREADS_PER_LOCALITY 4
    "-Ihpx.cds.num_concurrent_hazard_pointer_threads=256"
)

foreach(test ${tests})

  set(sources ${test}.cpp)

  source_group("Source Files" FILES ${sources})

  # add example executable
  add_hpx_executable(
    ${test}_test INTERNAL_FLAGS
    SOURCES ${sources} ${${test}_FLAGS}
    EXCLUDE_FROM_ALL
    DEPENDENCIES hpx_libcds
    FOLDER "Tests/Unit/Modules/LibCDS"
  )

  target_include_directories(${test}_test PRIVATE ${LIBCDS_SOURCE_DIR})

  add_hpx_unit_test("modules.libcds" ${test} ${${test}_PARAMETERS})

endforeach()

if(Boost_THREAD_FOUND)
  target_link_libraries(
    libcds_flat_combining_priority_queue_stdthread_test PUBLIC Boost::thread
  )
endif()
