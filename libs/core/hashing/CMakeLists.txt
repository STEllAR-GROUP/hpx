# Copyright (c) 2019 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(hashing_headers hpx/hashing/fibhash.hpp hpx/hashing/jenkins_hash.hpp)

# cmake-format: off
set(hashing_compat_headers
    hpx/hashing.hpp => hpx/modules/hashing.hpp
    hpx/util/fibhash.hpp => hpx/modules/hashing.hpp
    hpx/util/jenkins_hash.hpp => hpx/modules/hashing.hpp
)
# cmake-format: on

set(hashing_sources)

include(HPXLocal_AddModule)
hpx_local_add_module(
  local hashing
  GLOBAL_HEADER_GEN ON
  SOURCES ${hashing_sources}
  HEADERS ${hashing_headers}
  COMPAT_HEADERS ${hashing_compat_headers}
  MODULE_DEPENDENCIES hpx_config_local
  CMAKE_SUBDIRS examples tests
)
