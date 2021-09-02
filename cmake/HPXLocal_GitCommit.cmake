# Copyright (c) 2011 Bryce Lelbach
#               2015 Martin Stumpf
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(HPXLocal_GITCOMMIT_LOADED TRUE)

# if no git commit is set, try to get it from the source directory
if(NOT HPXLocal_WITH_GIT_COMMIT OR "${HPXLocal_WITH_GIT_COMMIT}" STREQUAL
                                   "None"
)

  find_package(Git)

  if(GIT_FOUND)
    execute_process(
      COMMAND "${GIT_EXECUTABLE}" "log" "--pretty=%H" "-1"
              "${PROJECT_SOURCE_DIR}"
      WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
      OUTPUT_VARIABLE HPXLocal_WITH_GIT_COMMIT
      ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE
    )
  endif()

endif()

if(NOT HPXLocal_WITH_GIT_COMMIT OR "${HPXLocal_WITH_GIT_COMMIT}" STREQUAL
                                   "None"
)
  hpx_local_warn("GIT commit not found (set to 'unknown').")
  set(HPXLocal_WITH_GIT_COMMIT "unknown")
  set(HPXLocal_WITH_GIT_COMMIT_SHORT "unknown")
else()
  hpx_local_info("GIT commit is ${HPXLocal_WITH_GIT_COMMIT}.")
  if(NOT HPXLocal_WITH_GIT_COMMIT_SHORT OR "${HPXLocal_WITH_GIT_COMMIT_SHORT}"
                                           STREQUAL "None"
  )
    string(SUBSTRING "${HPXLocal_WITH_GIT_COMMIT}" 0 7
                     HPXLocal_WITH_GIT_COMMIT_SHORT
    )
  endif()
endif()
