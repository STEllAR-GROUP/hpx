# Copyright (c) 2020 ETH Zurich
# Copyright (c) 2017 John Biddiscombe
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

cmake_minimum_required(VERSION 3.1 FATAL_ERROR)

set(CTEST_TEST_TIMEOUT 300)
set(CTEST_BUILD_PARALLELISM 20)
set(CTEST_TEST_PARALLELISM 4)
set(CTEST_CMAKE_GENERATOR Ninja)
set(CTEST_SITE "lsu(rostam)")
set(CTEST_UPDATE_COMMAND "git")
set(CTEST_UPDATE_VERSION_ONLY "ON")
set(CTEST_SUBMIT_RETRY_COUNT 5)
set(CTEST_SUBMIT_RETRY_DELAY 60)

if(NOT "$ENV{ghprbPullId}" STREQUAL "")
  set(CTEST_BUILD_NAME "$ENV{ghprbPullId}-${CTEST_BUILD_CONFIGURATION_NAME}")
  set(CTEST_TRACK "Pull_Requests")
else()
  set(CTEST_BUILD_NAME
      "$ENV{git_local_branch}-${CTEST_BUILD_CONFIGURATION_NAME}"
  )
  set(CTEST_TRACK "$ENV{git_local_branch}")
endif()

set(CTEST_CONFIGURE_COMMAND "${CMAKE_COMMAND} ${CTEST_SOURCE_DIRECTORY}")
set(CTEST_CONFIGURE_COMMAND
    "${CTEST_CONFIGURE_COMMAND} -G${CTEST_CMAKE_GENERATOR}"
)
set(CTEST_CONFIGURE_COMMAND
    "${CTEST_CONFIGURE_COMMAND} -B${CTEST_BINARY_DIRECTORY}"
)
set(CTEST_CONFIGURE_COMMAND
    "${CTEST_CONFIGURE_COMMAND} -DHPX_WITH_PARALLEL_TESTS_BIND_NONE=ON"
)
set(CTEST_CONFIGURE_COMMAND
    "${CTEST_CONFIGURE_COMMAND} ${CTEST_CONFIGURE_EXTRA_OPTIONS}"
)

ctest_start(Experimental TRACK "${CTEST_TRACK}")
ctest_update()
ctest_submit(PARTS Update)
ctest_configure()
ctest_submit(PARTS Configure)
ctest_build(TARGET all FLAGS "-k0 -j ${CTEST_BUILD_PARALLELISM}")
ctest_build(TARGET tests FLAGS "-k0 -j ${CTEST_BUILD_PARALLELISM}")
ctest_submit(PARTS Build)
ctest_test(PARALLEL_LEVEL "${CTEST_TEST_PARALLELISM}")
ctest_submit(PARTS Test BUILD_ID CTEST_BUILD_ID)
file(WRITE "jenkins-hpx-${CTEST_BUILD_CONFIGURATION_NAME}-cdash-build-id.txt"
     "${CTEST_BUILD_ID}"
)
