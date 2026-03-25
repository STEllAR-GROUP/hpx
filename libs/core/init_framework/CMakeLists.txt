# Copyright (c) 2026 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(init_framework_headers
    hpx/init_params.hpp
)

set(init_framework_sources
    src/init_framework.cpp
)

include(HPX_AddModule)
add_hpx_module(
  core init_framework
  GLOBAL_HEADER_GEN OFF
  HEADERS ${init_framework_headers}
  SOURCES ${init_framework_sources}
  DEPENDENCIES hpx_core
  MODULE_DEPENDENCIES hpx_program_options hpx_runtime_local hpx_functional
                      hpx_init_runtime_local hpx_runtime_configuration
)
