# Copyright (c) 2019-2023 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

# Allow to disable caching allocator
hpx_option(
  HPX_ALLOCATOR_SUPPORT_WITH_CACHING BOOL
  "Enable caching allocator. (default: ON)" ON ADVANCED
  CATEGORY "Modules"
  MODULE ALLOCATOR_SUPPORT
)

if(HPX_ALLOCATOR_SUPPORT_WITH_CACHING)
  hpx_add_config_define_namespace(
    DEFINE HPX_ALLOCATOR_SUPPORT_HAVE_CACHING NAMESPACE ALLOCATOR_SUPPORT
  )
endif()

set(allocator_support_headers
    hpx/allocator_support/aligned_allocator.hpp
    hpx/allocator_support/allocator_deleter.hpp
    hpx/allocator_support/detail/new.hpp
    hpx/allocator_support/internal_allocator.hpp
    hpx/allocator_support/traits/is_allocator.hpp
)

if(HPX_ALLOCATOR_SUPPORT_WITH_CACHING)
  set(allocator_support_headers
      ${allocator_support_headers}
      hpx/allocator_support/thread_local_caching_allocator.hpp
  )
endif()

# cmake-format: off
set(allocator_support_compat_headers
    hpx/allocator_support.hpp => hpx/modules/allocator_support.hpp
    hpx/util/allocator_deleter.hpp => hpx/modules/allocator_support.hpp
    hpx/util/internal_allocator.hpp => hpx/modules/allocator_support.hpp
    hpx/compute/detail/new.hpp => hpx/modules/allocator_support.hpp
)
# cmake-format: on

set(allocator_support_sources)

include(HPX_AddModule)
add_hpx_module(
  core allocator_support
  GLOBAL_HEADER_GEN ON
  SOURCES ${allocator_support_sources}
  HEADERS ${allocator_support_headers}
  COMPAT_HEADERS ${allocator_support_compat_headers}
  DEPENDENCIES hpx_dependencies_allocator
  MODULE_DEPENDENCIES hpx_concepts hpx_config hpx_preprocessor hpx_type_support
  CMAKE_SUBDIRS examples tests
)
