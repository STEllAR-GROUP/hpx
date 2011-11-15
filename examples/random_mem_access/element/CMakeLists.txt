# Copyright (c) 2007-2009 Hartmut Kaiser
# Copyright (c) 2011 Matt Anderson
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

###############################################################################
set(client_sources
    element.cpp)

set(server_sources
    server/element.cpp)

###############################################################################
set(client_headers
    element.hpp)

set(stubs_headers
    stubs/element.hpp)

set(server_headers
    server/element.hpp)

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
add_hpx_component(random_mem_access_element
  MODULE random_mem_access
  SOURCES ${client_sources} ${server_sources}
  HEADERS ${headers})

