# Copyright (c) 2020 Kor de Jong
# Copyright (c) 2020 ETH Zurich
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

# This project tests that HPX can be built using FetchContent. It is built by
# the .github/workflows/linux_fetchcontent.yml GitHub actions workflow.

cmake_minimum_required(VERSION 3.13)
project(HPXFetchContent LANGUAGES CXX)

# Test that HPX correctly handles having Boost Filesystem and ProgramOptions
# found before setting up HPX, if they are available (do not REQUIRE these)
find_package(Boost COMPONENTS filesystem program_options)

include(FetchContent)

fetchcontent_declare(
  hpx
  GIT_REPOSITORY ${HPX_REPOSITORY}
  GIT_TAG ${HPX_TAG}
)
fetchcontent_getproperties(hpx)
if(NOT hpx_POPULATED)
  fetchcontent_populate(hpx)
  add_subdirectory(${hpx_SOURCE_DIR} ${hpx_BINARY_DIR})
endif()
