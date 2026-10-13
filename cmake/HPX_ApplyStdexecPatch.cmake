#  Copyright (c) 2026 Hartmut Kaiser
#  Copyright (c) 2026 Anshuman Agrawal
#
#  SPDX-License-Identifier: BSL-1.0
#  Distributed under the Boost Software License, Version 1.0. (See accompanying
#  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

# Applies a required compatibility patch to the fetched stdexec sources.

if(NOT DEFINED HPX_STDEXEC_SOURCE_DIR)
  message(FATAL_ERROR "HPX_STDEXEC_SOURCE_DIR must be defined")
endif()

if(NOT DEFINED HPX_STDEXEC_PATCH_FILE)
  message(FATAL_ERROR "HPX_STDEXEC_PATCH_FILE must be defined")
endif()

find_package(Git QUIET)
if(NOT GIT_EXECUTABLE)
  message(FATAL_ERROR "git not found, cannot patch the fetched stdexec sources")
endif()

# Already applied (a reconfigure re-runs the patch command)?
execute_process(
  COMMAND "${GIT_EXECUTABLE}" apply --reverse --check
          "${HPX_STDEXEC_PATCH_FILE}"
  WORKING_DIRECTORY "${HPX_STDEXEC_SOURCE_DIR}"
  RESULT_VARIABLE _hpx_stdexec_patch_applied
  OUTPUT_QUIET ERROR_QUIET
)
if(_hpx_stdexec_patch_applied EQUAL 0)
  return()
endif()

execute_process(
  COMMAND "${GIT_EXECUTABLE}" apply "${HPX_STDEXEC_PATCH_FILE}"
  WORKING_DIRECTORY "${HPX_STDEXEC_SOURCE_DIR}"
  RESULT_VARIABLE _hpx_stdexec_patch_result
  ERROR_VARIABLE _hpx_stdexec_patch_error
)
if(NOT _hpx_stdexec_patch_result EQUAL 0)
  message(
    FATAL_ERROR
      "Failed to patch the fetched stdexec sources: ${_hpx_stdexec_patch_error}"
  )
endif()
