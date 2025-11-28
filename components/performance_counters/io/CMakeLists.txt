# Copyright (c) 2007-2023 Hartmut Kaiser
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

if(HPX_WITH_IO_COUNTERS)
  set(HPX_COMPONENTS
      ${HPX_COMPONENTS} io_counters
      CACHE INTERNAL "list of HPX components"
  )

  set(io_counters_headers
      hpx/components/performance_counters/io/io_counters.hpp
  )

  set(io_counters_sources io_counters.cpp)

  add_hpx_component(
    io_counters INTERNAL_FLAGS
    FOLDER "Core/Components/Counters"
    INSTALL_HEADERS PLUGIN PREPEND_HEADER_ROOT
    INSTALL_COMPONENT runtime
    HEADER_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/include"
    HEADERS ${io_counters_headers}
    PREPEND_SOURCE_ROOT ${HPX_WITH_UNITY_BUILD_OPTION}
    ${HPX_WITH_CXX_MODULES_OPTION}
    SOURCE_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/src"
    SOURCES ${io_counters_sources}
  )

  # see #6579
  if(MSVC AND Boost_VERSION_STRING VERSION_LESS "1.79.0")
    target_compile_definitions(io_counters_component PRIVATE NOMINMAX)
  endif()

  add_hpx_pseudo_dependencies(
    components.performance_counters.io io_counters_component
  )

  add_subdirectory(tests)
  add_subdirectory(examples)
endif()
