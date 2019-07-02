# Copyright (c) 2007-2012 Hartmut Kaiser
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

if(HPX_WITH_PAPI)
  find_package(PAPI)

  if(NOT PAPI_FOUND)
    hpx_error("PAPI could not be found and HPX_WITH_PAPI=On, please specify PAPI_ROOT to point to the root of your PAPI installation")
  endif()

  include_directories(${PAPI_INCLUDE_DIR})

  if(NOT HPX_WITH_DEFAULT_TARGETS)
    set(_exclude_from_all_flag EXCLUDE_FROM_ALL)
  endif()

  set(papi_counters_headers
    hpx/components/performance_counters/papi/server/papi.hpp
    hpx/components/performance_counters/papi/util/papi.hpp
  )

  set(papi_counters_sources
    server/papi.cpp
    util/papi.cpp
  )

  add_hpx_component(papi_counters
    FOLDER "Core/Components/IO"
    INSTALL_HEADERS
    HEADER_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/include"
    HEADERS ${papi_counters_headers}
    SOURCE_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/src"
    SOURCES ${papi_counters_sources}
    ${_exclude_from_all_flag})

  add_hpx_pseudo_dependencies(components.performance_counters.papi_counters papi_counters_component)
endif()
