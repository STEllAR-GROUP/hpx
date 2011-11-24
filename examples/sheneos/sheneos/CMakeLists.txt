# Copyright (c) 2007-2011 Hartmut Kaiser
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

###############################################################################
set(client_sources
    dimension.cpp
    interpolator.cpp
    read_values.cpp)

set(server_sources
    server/configuration.cpp
    server/partition3d.cpp)

###############################################################################
set(client_headers
    configuration.hpp
    dimension.hpp
    interpolator.hpp
    partition3d.hpp
    read_values.hpp)

set(stubs_headers
    stubs/configuration.hpp
    stubs/partition3d.hpp)

set(server_headers
    server/configuration.hpp
    server/partition3d.hpp)

set(headers
    ${client_headers}
    ${stubs_headers}
    ${server_headers})

###############################################################################
set(dependencies
    ${HDF5_CPP_LIBRARY}
    distributing_factory_component)

###############################################################################
source_group("Source Files\\Client" FILES ${client_sources})
source_group("Source Files\\Server" FILES ${server_sources})
source_group("Header Files\\Client" FILES ${client_headers})
source_group("Header Files\\Stubs"  FILES ${stubs_headers})
source_group("Header Files\\Server" FILES ${server_headers})

###############################################################################
add_hpx_component(sheneos
  SOURCES ${client_sources} ${server_sources}
  HEADERS ${headers}
  DEPENDENCIES ${dependencies})

