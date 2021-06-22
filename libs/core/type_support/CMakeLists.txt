# Copyright (c) 2019 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(type_support_headers
    hpx/type_support/detected.hpp hpx/type_support/equality.hpp
    hpx/type_support/identity.hpp hpx/type_support/static.hpp
    hpx/type_support/unused.hpp
)

# cmake-format: off
set(type_support_compat_headers
    hpx/type_support.hpp => hpx/modules/type_support.hpp
    hpx/util/detected.hpp => hpx/modules/type_support.hpp
    hpx/util/identity.hpp => hpx/modules/type_support.hpp
    hpx/util/static.hpp => hpx/modules/type_support.hpp
    hpx/util/unused.hpp => hpx/modules/type_support.hpp
)
# cmake-format: on

set(type_support_sources)

include(HPX_AddModule)
add_hpx_module(
  core type_support
  GLOBAL_HEADER_GEN ON
  SOURCES ${type_support_sources}
  HEADERS ${type_support_headers}
  COMPAT_HEADERS ${type_support_compat_headers}
  MODULE_DEPENDENCIES hpx_assertion hpx_config hpx_fundamentals
  CMAKE_SUBDIRS examples tests
)
