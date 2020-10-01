# Copyright (c) 2019 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

cmake_minimum_required(VERSION 3.13 FATAL_ERROR)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# Default location is $HPX_ROOT/libs/plugin/include
set(plugin_headers
    hpx/modules/plugin.hpp
    hpx/plugin/abstract_factory.hpp
    hpx/plugin/concrete_factory.hpp
    hpx/plugin/config.hpp
    hpx/plugin/dll.hpp
    hpx/plugin/export_plugin.hpp
    hpx/plugin/plugin_factory.hpp
    hpx/plugin/plugin_wrapper.hpp
    hpx/plugin/virtual_constructor.hpp
    hpx/plugin/detail/dll_dlopen.hpp
    hpx/plugin/detail/dll_windows.hpp
    hpx/plugin/traits/plugin_config_data.hpp
)

# Default location is $HPX_ROOT/libs/plugin/include_compatibility
# cmake-format: off
set(plugin_compat_headers
    hpx/plugin.hpp => hpx/modules/plugin.hpp
    hpx/util/plugin.hpp => hpx/modules/plugin.hpp
    hpx/util/plugin/abstract_factory.hpp => hpx/modules/plugin.hpp
    hpx/util/plugin/concrete_factory.hpp => hpx/modules/plugin.hpp
    hpx/util/plugin/config.hpp => hpx/modules/plugin.hpp
    hpx/util/plugin/dll.hpp => hpx/modules/plugin.hpp
    hpx/util/plugin/export_plugin.hpp => hpx/modules/plugin.hpp
    hpx/util/plugin/plugin_factory.hpp => hpx/modules/plugin.hpp
    hpx/util/plugin/plugin_wrapper.hpp => hpx/modules/plugin.hpp
    hpx/util/plugin/virtual_constructor.hpp => hpx/modules/plugin.hpp
    hpx/traits/plugin_config_data.hpp => hpx/modules/plugin.hpp
)
# cmake-format: on

include(HPX_AddModule)
add_hpx_module(
  core plugin
  COMPATIBILITY_HEADERS ON # Added in 1.5.0
  DEPRECATION_WARNINGS
  GLOBAL_HEADER_GEN OFF
  HEADERS ${plugin_headers}
  COMPAT_HEADERS ${plugin_compat_headers}
  MODULE_DEPENDENCIES hpx_assertion hpx_config hpx_errors hpx_datastructures
                      hpx_filesystem hpx_functional
  CMAKE_SUBDIRS tests
)
