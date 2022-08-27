# Copyright (c) 2019-2021 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(lock_registration_headers hpx/lock_registration/detail/register_locks.hpp)
set(lock_registration_sources register_locks.cpp)

include(HPX_AddModule)
add_hpx_module(
  core lock_registration
  GLOBAL_HEADER_GEN ON
  SOURCES ${lock_registration_sources}
  HEADERS ${lock_registration_headers}
  DEPENDENCIES hpx_assertion hpx_concepts hpx_config hpx_errors hpx_functional
               hpx_type_support
  CMAKE_SUBDIRS examples tests
)
