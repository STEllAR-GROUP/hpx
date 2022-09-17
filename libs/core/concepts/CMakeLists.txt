# Copyright (c) 2019 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(concepts_headers hpx/concepts/concepts.hpp hpx/concepts/has_member_xxx.hpp
                     hpx/concepts/has_xxx.hpp
)

# cmake-format: off
set(concepts_compat_headers
    hpx/concepts.hpp => hpx/modules/concepts.hpp
    hpx/traits/concepts.hpp => hpx/modules/concepts.hpp
    hpx/traits/has_member_xxx.hpp => hpx/modules/concepts.hpp
    hpx/traits/has_xxx.hpp => hpx/modules/concepts.hpp
)
# cmake-format: on

set(concepts_sources)

if(HPX_WITH_CXX20_MODULES)
  set(concepts_modulefiles hpx.core.concepts.cppm)
endif()

include(HPX_AddModule)
add_hpx_module(
  core concepts
  GLOBAL_HEADER_GEN MODULE
  SOURCES ${concepts_sources}
  HEADERS ${concepts_headers}
  MODULES ${concepts_modulefiles}
  COMPAT_HEADERS ${concepts_compat_headers}
  MODULE_DEPENDENCIES hpx_config hpx_preprocessor
  CMAKE_SUBDIRS examples tests
)
