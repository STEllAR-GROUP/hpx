# Copyright (c) 2019-2020 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

cmake_minimum_required(VERSION 3.13 FATAL_ERROR)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(runtime_configuration_headers
    hpx/runtime_configuration/agas_service_mode.hpp
    hpx/runtime_configuration/component_registry_base.hpp
    hpx/runtime_configuration/ini.hpp
    hpx/runtime_configuration/init_ini_data.hpp
    hpx/runtime_configuration/plugin_registry_base.hpp
    hpx/runtime_configuration/runtime_configuration.hpp
    hpx/runtime_configuration/runtime_configuration_fwd.hpp
    hpx/runtime_configuration/runtime_mode.hpp
    hpx/runtime_configuration/static_factory_data.hpp
)

set(runtime_configuration_compat_headers
    hpx/util/ini.hpp
    hpx/util/init_ini_data.hpp
    hpx/util/runtime_configuration.hpp
    hpx/plugins/plugin_registry_base.hpp
    hpx/runtime/components/component_registry_base.hpp
    hpx/runtime/components/static_factory_data.hpp
    hpx/runtime/runtime_mode.hpp
)

set(runtime_configuration_sources ini.cpp init_ini_data.cpp
                                  runtime_configuration.cpp runtime_mode.cpp
)

include(HPX_AddModule)
add_hpx_module(
  full runtime_configuration
  COMPATIBILITY_HEADERS ON
  DEPRECATION_WARNINGS
  GLOBAL_HEADER_GEN ON
  SOURCES ${runtime_configuration_sources}
  HEADERS ${runtime_configuration_headers}
  COMPAT_HEADERS ${runtime_configuration_compat_headers}
  DEPENDENCIES hpx_core hpx_parallelism
  CMAKE_SUBDIRS examples tests
)
