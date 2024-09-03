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

ctest_start(Experimental TRACK "${CTEST_TRACK}")

ctest_update()
ctest_submit(
  PARTS Update
  BUILD_ID __ctest_build_id
  RETURN_VALUE __update_result
)

set(CTEST_CONFIGURE_COMMAND "${CMAKE_COMMAND} ${CTEST_SOURCE_DIRECTORY}")
set(CTEST_CONFIGURE_COMMAND
    "${CTEST_CONFIGURE_COMMAND} -G${CTEST_CMAKE_GENERATOR}"
)
set(CTEST_CONFIGURE_COMMAND
    "${CTEST_CONFIGURE_COMMAND} -B${CTEST_BINARY_DIRECTORY}"
)
set(CTEST_CONFIGURE_COMMAND
    "${CTEST_CONFIGURE_COMMAND} -DHPX_WITH_NANOBENCH=ON"
)
set(CTEST_CONFIGURE_COMMAND
    "${CTEST_CONFIGURE_COMMAND} -DHPX_WITH_PARALLEL_TESTS_BIND_NONE=ON"
)
set(CTEST_CONFIGURE_COMMAND
    "${CTEST_CONFIGURE_COMMAND} ${CTEST_CONFIGURE_EXTRA_OPTIONS}"
)

ctest_configure()
ctest_submit(
  PARTS Configure
  BUILD_ID __ctest_build_id
  RETURN_VALUE __configure_result
)
if(NOT CTEST_BUILD_ID AND __ctest_build_id)
  set(CTEST_BUILD_ID ${__ctest_build_id})
endif()
set(ctest_submission_result ${ctest_submission_result} "Configure: "
                            ${__configure_result} "\n"
)

set(benchmarks
  minmax_element_performance
  small_vector_benchmark
  future_overhead_report
  stream_report
  foreach_report
  transform_reduce_scaling
  benchmark_is_heap_until
  benchmark_merge
  benchmark_inplace_merge
  benchmark_is_heap
  benchmark_remove
  benchmark_remove_if
  benchmark_partial_sort
  benchmark_partial_sort_parallel
  benchmark_nth_element
  benchmark_nth_element_parallel
)

foreach(benchmark ${benchmarks})
  ctest_build(TARGET ${benchmark}_cdash_results FLAGS "-k0 -j ${CTEST_BUILD_PARALLELISM}")
endforeach()

ctest_submit(
  PARTS Build
  BUILD_ID __ctest_build_id
  RETURN_VALUE __build_result
)
if(NOT CTEST_BUILD_ID AND __ctest_build_id)
  set(CTEST_BUILD_ID ${__ctest_build_id})
endif()
set(ctest_submission_result ${ctest_submission_result} "Build: "
                            ${__build_result} "\n"
)

ctest_test(
  INCLUDE "_perftest$"
  PARALLEL_LEVEL "${CTEST_TEST_PARALLELISM}"
)
ctest_submit(
  PARTS Test
  BUILD_ID __ctest_build_id
  RETURN_VALUE __test_result
)

if(NOT CTEST_BUILD_ID AND __ctest_build_id)
  set(CTEST_BUILD_ID ${__ctest_build_id})
endif()
set(ctest_submission_result ${ctest_submission_result} "Tests: "
                            ${__test_result} "\n"
)

file(WRITE "jenkins-hpx-${CTEST_BUILD_CONFIGURATION_NAME}-cdash-build-id.txt"
     "${CTEST_BUILD_ID}"
)
file(WRITE "jenkins-hpx-${CTEST_BUILD_CONFIGURATION_NAME}-cdash-submission.txt"
     ${ctest_submission_result}
)