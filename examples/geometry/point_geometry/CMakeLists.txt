# Copyright (c) 2007-2012 Hartmut Kaiser
# Copyright (c) 2011 Matt Anderson
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

###############################################################################
set(client_sources
    point.cpp
    serialize_geometry.cpp)

###############################################################################
set(common_headers
    serialize_geometry.hpp)

set(client_headers
    point.hpp)

set(stubs_headers
    stubs/point.hpp)

set(server_headers
    server/point.hpp)

set(server_sources
    server/point.cpp)

###############################################################################
set(sources
    ${client_sources}
    ${server_sources})

set(headers
    ${common_headers}
    ${client_headers}
    ${stubs_headers}
    ${server_headers})

###############################################################################
source_group("Source Files\\Client" FILES ${client_sources})
source_group("Source Files\\Client" FILES ${server_sources})
source_group("Header Files\\Client" FILES ${client_headers})
source_group("Header Files\\Stubs"  FILES ${stubs_headers})
source_group("Header Files\\Server" FILES ${server_headers})
source_group("Header Files" FILES ${common_headers})

###############################################################################
add_hpx_component(point_geometry
    MODULE point_geometry
    SOURCES ${sources}
    HEADERS ${headers}
    FOLDER "Examples/Geometry/point_geometry")

