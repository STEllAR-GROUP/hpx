# Copyright (c) 2007-2012 Hartmut Kaiser
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

###############################################################################
set(client_sources
    simple_accumulator.cpp)

###############################################################################
set(client_headers
    simple_accumulator.hpp)

set(stubs_headers
    stubs/simple_accumulator.hpp)

set(server_headers
    server/simple_accumulator.hpp)

###############################################################################
set(headers
    ${client_headers}
    ${stubs_headers}
    ${server_headers})

###############################################################################
source_group("Source Files\\Client" FILES ${client_sources})
source_group("Header Files\\Client" FILES ${client_headers})
source_group("Header Files\\Stubs"  FILES ${stubs_headers})
source_group("Header Files\\Server" FILES ${server_headers})

###############################################################################
add_hpx_component(simple_accumulator
  SOURCES ${client_sources}
  HEADERS ${headers}
  FOLDER "Examples/Quickstart/simple_accumulator")

