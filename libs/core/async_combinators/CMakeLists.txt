# Copyright (c) 2019 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(async_combinators_headers
    hpx/async_combinators/future_wait.hpp
    hpx/async_combinators/split_future.hpp
    hpx/async_combinators/wait_all.hpp
    hpx/async_combinators/wait_any.hpp
    hpx/async_combinators/wait_each.hpp
    hpx/async_combinators/wait_some.hpp
    hpx/async_combinators/when_all.hpp
    hpx/async_combinators/when_any.hpp
    hpx/async_combinators/when_each.hpp
    hpx/async_combinators/when_some.hpp
)

# cmake-format: off
set(async_combinators_compat_headers
    hpx/lcos/future_wait.hpp => hpx/future.hpp
    hpx/lcos/split_future.hpp => hpx/future.hpp
    hpx/lcos/wait_all.hpp => hpx/future.hpp
    hpx/lcos/wait_any.hpp => hpx/future.hpp
    hpx/lcos/wait_each.hpp => hpx/future.hpp
    hpx/lcos/wait_some.hpp => hpx/future.hpp
    hpx/lcos/when_all.hpp => hpx/future.hpp
    hpx/lcos/when_any.hpp => hpx/future.hpp
    hpx/lcos/when_each.hpp => hpx/future.hpp
    hpx/lcos/when_some.hpp => hpx/future.hpp
)
# cmake-format: on

include(HPXLocal_AddModule)
hpx_local_add_module(
  local async_combinators
  GLOBAL_HEADER_GEN ON
  HEADERS ${async_combinators_headers}
  COMPAT_HEADERS ${async_combinators_compat_headers}
  MODULE_DEPENDENCIES hpx_async_base hpx_config_local hpx_errors hpx_futures
                      hpx_memory hpx_pack_traversal
  CMAKE_SUBDIRS examples tests
)
