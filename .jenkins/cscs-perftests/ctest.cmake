# Copyright (c) 2020 ETH Zurich
# Copyright (c) 2017 John Biddiscombe
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

# This is a dummy file to trigger the upload of the perftests reports
cmake_minimum_required(VERSION 3.1 FATAL_ERROR)

set(CTEST_TEST_TIMEOUT 300)
set(CTEST_BUILD_PARALLELISM 20)
set(CTEST_SITE "cscs(daint)")
set(CTEST_UPDATE_COMMAND "git")
set(CTEST_UPDATE_VERSION_ONLY "ON")
set(CTEST_SUBMIT_RETRY_COUNT 5)
set(CTEST_SUBMIT_RETRY_DELAY 60)

if(NOT "$ENV{ghprbPullId}" STREQUAL "")
  set(CTEST_BUILD_NAME "$ENV{ghprbPullId}-${CTEST_BUILD_CONFIGURATION_NAME}")
  set(CTEST_TRACK "Experimental")
else()
  set(CTEST_BUILD_NAME
      "$ENV{git_local_branch}-${CTEST_BUILD_CONFIGURATION_NAME}"
  )
  set(CTEST_TRACK "$ENV{git_local_branch}")
endif()

ctest_start(Experimental TRACK "${CTEST_TRACK}")
ctest_update()
