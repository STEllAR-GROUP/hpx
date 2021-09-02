# Copyright (c) 2019-2020 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(runtime_configuration_headers
    hpx/runtime_configuration/agas_service_mode.hpp
    hpx/runtime_configuration/component_commandline_base.hpp
    hpx/runtime_configuration/component_factory_base.hpp
    hpx/runtime_configuration/component_registry_base.hpp
    hpx/runtime_configuration/init_ini_data.hpp
    hpx/runtime_configuration/plugin_registry_base.hpp
    hpx/runtime_configuration/register_locks_globally.hpp
    hpx/runtime_configuration/runtime_configuration.hpp
    hpx/runtime_configuration/runtime_configuration_fwd.hpp
    hpx/runtime_configuration/runtime_mode.hpp
    hpx/runtime_configuration/static_factory_data.hpp
)

# cmake-format: off
set(
  runtime_configuration_compat_headers
  hpx/util/ini.hpp => hpx/modules/runtime_configuration.hpp
  hpx/util/init_ini_data.hpp => hpx/modules/runtime_configuration.hpp
  hpx/util/register_locks_globally.hpp => hpx/modules/runtime_configuration.hpp
  hpx/util/runtime_configuration.hpp => hpx/modules/runtime_configuration.hpp
  hpx/plugins/plugin_registry_base.hpp => hpx/modules/runtime_configuration.hpp
  hpx/runtime/components/component_commandline_base.hpp => hpx/modules/runtime_configuration.hpp
  hpx/runtime/components/component_factory_base.hpp => hpx/modules/runtime_configuration.hpp
  hpx/runtime/components/component_registry_base.hpp => hpx/modules/runtime_configuration.hpp
  hpx/runtime/components/static_factory_data.hpp => hpx/modules/runtime_configuration.hpp
  hpx/runtime/runtime_mode.hpp => hpx/modules/runtime_configuration.hpp
)
# cmake-format: on

set(runtime_configuration_sources
    init_ini_data.cpp register_locks_globally.cpp runtime_configuration.cpp
    runtime_mode.cpp static_factory_data.cpp
)

include(HPXLocal_AddModule)
hpx_local_add_module(
  local runtime_configuration
  GLOBAL_HEADER_GEN ON
  SOURCES ${runtime_configuration_sources}
  HEADERS ${runtime_configuration_headers}
  COMPAT_HEADERS ${runtime_configuration_compat_headers}
  MODULE_DEPENDENCIES
    hpx_program_options
    hpx_errors
    hpx_filesystem
    hpx_itt_notify
    hpx_logging
    hpx_plugin
    hpx_ini
    hpx_prefix
    hpx_coroutines
    hpx_version_local
    hpx_synchronization
  CMAKE_SUBDIRS examples tests
)
