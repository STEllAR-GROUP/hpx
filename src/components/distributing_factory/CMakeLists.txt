# Copyright (c) 2007-2009 Hartmut Kaiser
# Copyright (c)      2011 Bryce Lelbach
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

###############################################################################
set(root ${hpx_SOURCE_DIR}/hpx/components/distributing_factory)

set(client_headers ${root}/distributing_factory.hpp)

set(stubs_headers ${root}/stubs/distributing_factory.hpp)

set(server_headers ${root}/server/distributing_factory.hpp)

###############################################################################
set(client_sources
    distributing_factory.cpp)

set(server_sources
    server/distributing_factory.cpp)

###############################################################################
set(sources
    ${client_sources}
    ${server_sources})

###############################################################################
set(headers
    ${client_headers}
    ${stubs_headers}
    ${server_headers})

###############################################################################
source_group("Source Files\\Client" FILES ${client_sources})
source_group("Source Files\\Server" FILES ${server_sources})
source_group("Header Files\\Client" FILES ${client_headers})
source_group("Header Files\\Stubs"  FILES ${stubs_headers})
source_group("Header Files\\Server" FILES ${server_headers})

###############################################################################
add_hpx_component(distributing_factory
  SOURCES ${sources}
  HEADERS ${headers}
  FOLDER "Core")

