# Copyright (c) 2007-2022 Hartmut Kaiser
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

if(NOT HPX_WITH_DISTRIBUTED_RUNTIME)
  return()
endif()

# The Power counters depend on th PWR library (http://powerapi.sandia.gov/)
find_package(PWR QUIET)
if(NOT PWR_FOUND)
  return()
endif()

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
  HEADER_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/include"
  HEADERS ${power_headers}
  PREPEND_SOURCE_ROOT
  SOURCE_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/src"
  SOURCES ${power_sources} ${HPX_WITH_UNITY_BUILD_OPTION}
  DEPENDENCIES PWR::pwr
)

add_hpx_pseudo_dependencies(
  components.performance_counters.power power_component
)
