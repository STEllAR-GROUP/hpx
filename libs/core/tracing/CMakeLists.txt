# Copyright (c) 2026 Vansh Dobhal
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(tracing_headers hpx/tracing/tracing.hpp)
set(tracing_sources)

set(tracing_module_dependencies hpx_config hpx_threading_base)
if(HPX_TRACY_WITH_TRACY)
  list(APPEND tracing_module_dependencies hpx_tracy)
endif()

include(HPX_AddModule)
add_hpx_module(
  core tracing
  GLOBAL_HEADER_GEN ON
  GLOBAL_HEADER_MODULE_GEN ON
  SOURCES ${tracing_sources}
  HEADERS ${tracing_headers}
  MODULE_DEPENDENCIES ${tracing_module_dependencies}
)
