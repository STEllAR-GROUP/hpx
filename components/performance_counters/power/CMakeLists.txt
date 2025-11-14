# Copyright (c) 2007-2023 Hartmut Kaiser
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

if(NOT HPX_WITH_DISTRIBUTED_RUNTIME)
  return()
endif()

hpx_option(
  HPX_WITH_POWER_COUNTER BOOL
  "Enable use of performance counters based on pwr library (default: OFF)" OFF
  ADVANCED
  CATEGORY "Modules"
  MODULE POWER_COUNTER
)

if(NOT HPX_WITH_POWER_COUNTER)
  return()
endif()

# The Power counters depend on the PWR library (http://powerapi.sandia.gov/)
find_package(PWR QUIET)
if(NOT PWR_FOUND)
  hpx_warning("PWR library was not found, setting HPX_WITH_POWER_COUNTER=OFF")
  hpx_set_option(
    HPX_WITH_POWER_COUNTER
    VALUE OFF
    FORCE
  )
  return()
endif()

hpx_add_config_define(HPX_HAVE_POWER_COUNTER)

set(HPX_COMPONENTS
    ${HPX_COMPONENTS} power
    CACHE INTERNAL "list of HPX components"
)

set(power_headers hpx/components/performance_counters/power/power_counter.hpp)

set(power_sources power.cpp power_counter.cpp)

add_hpx_component(
  power INTERNAL_FLAGS
  FOLDER "Core/Components/Counters"
  INSTALL_HEADERS PLUGIN PREPEND_HEADER_ROOT
  INSTALL_COMPONENT runtime
  HEADER_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/include"
  HEADERS ${power_headers}
  PREPEND_SOURCE_ROOT ${HPX_WITH_UNITY_BUILD_OPTION}
  ${HPX_WITH_CXX_MODULES_OPTION}
  SOURCE_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/src"
  SOURCES ${power_sources}
  DEPENDENCIES PWR::pwr
)

add_hpx_pseudo_dependencies(
  components.performance_counters.power power_component
)

add_subdirectory(tests)
