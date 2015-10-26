# Copyright (c) 2011 Bryce Lelbach
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(HPX_GITCOMMIT_LOADED TRUE)

find_package(Git)

if(GIT_FOUND)
  execute_process(
    COMMAND "${GIT_EXECUTABLE}" "log" "--pretty=%H" "-1" "${PROJECT_SOURCE_DIR}"
    WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
    OUTPUT_VARIABLE HPX_WITH_GIT_COMMIT ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)
endif()

if(NOT GIT_FOUND OR NOT HPX_WITH_GIT_COMMIT OR "${HPX_WITH_GIT_COMMIT}" STREQUAL "None")
  hpx_error("GIT commit not found (set to 'unknown').")
  set(HPX_WITH_GIT_COMMIT "unknown")
else()
  hpx_error("GIT commit is ${HPX_WITH_GIT_COMMIT}.")
endif()

