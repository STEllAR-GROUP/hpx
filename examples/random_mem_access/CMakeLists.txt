# Copyright (c) 2011      Matt Anderson
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

if(NOT HPX_WITH_DISTRIBUTED_RUNTIME)
  return()
endif()

# just recurse into all components subdirectories and execute the scripts there
set(subdirs random_mem_access)

set(random_mem_access_PARAMETERS THREADS_PER_LOCALITY 4)

# for all targets specified above
foreach(subdir ${subdirs})
  add_subdirectory(${subdir})

  set(sources ${subdir}_client.cpp)

  source_group("Source Files" FILES ${sources})

  # add example executable
  add_hpx_executable(
    ${subdir}_client INTERNAL_FLAGS MODULE random_mem_access
    SOURCES ${sources}
    DEPENDENCIES ${subdir}_component
    FOLDER "Examples/RandomMemoryAccess/random_mem_access"
  )

  add_hpx_example_target_dependencies("random_mem_access" ${subdir}_client)

  if(HPX_WITH_TESTS AND HPX_WITH_TESTS_EXAMPLES)
    add_hpx_example_test(
      "random_mem_access" ${subdir}_client ${${subdir}_PARAMETERS}
    )
  endif()

endforeach()
