# Copyright (c) 2007-2012 Hartmut Kaiser
# Copyright (c)      2011 Bryce Lelbach
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

if(NOT HPX_WITH_DISTRIBUTED_RUNTIME)
  return()
endif()

set(HPX_COMPONENTS
    ${HPX_COMPONENTS} iostreams
    CACHE INTERNAL "list of HPX components"
)

if(NOT HPX_WITH_DEFAULT_TARGETS)
  set(_exclude_from_all_flag EXCLUDE_FROM_ALL)
endif()

set(iostreams_headers
    hpx/components/iostreams/server/buffer.hpp
    hpx/components/iostreams/server/order_output.hpp
    hpx/components/iostreams/server/output_stream.hpp
    hpx/components/iostreams/export_definitions.hpp
    hpx/components/iostreams/manipulators.hpp
    hpx/components/iostreams/ostream.hpp
    hpx/components/iostreams/standard_streams.hpp
    hpx/components/iostreams/write_functions.hpp
    hpx/iostream.hpp
    hpx/include/iostreams.hpp
)

set(iostreams_sources server/output_stream.cpp component_module.cpp
                      manipulators.cpp standard_streams.cpp
)

add_hpx_component(
  iostreams INTERNAL_FLAGS
  FOLDER "Core/Components/IO"
  INSTALL_HEADERS PREPEND_HEADER_ROOT
  HEADER_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/include"
  HEADERS ${iostreams_headers}
  PREPEND_SOURCE_ROOT
  SOURCE_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/src"
  SOURCES ${iostreams_sources} ${_exclude_from_all_flag}
          ${HPX_WITH_UNITY_BUILD_OPTION}
)

target_compile_definitions(
  iostreams_component PRIVATE HPX_IOSTREAMS_MODULE_EXPORTS
)

add_hpx_pseudo_dependencies(components.iostreams iostreams_component)

add_subdirectory(tests)
add_subdirectory(examples)
