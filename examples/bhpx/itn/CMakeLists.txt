# Copyright (c) 2007-2011 Chirag Dekate
# Copyright (c) 2011      Bryce Lelbach
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying 
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

###############################################################################
set(client_sources
    itn.cpp)

set(server_sources
    server/itn.cpp)

###############################################################################
set(client_headers
    itn.hpp)

set(stubs_headers
    stubs/itn.hpp)

set(server_headers
    server/itn.hpp)

###############################################################################
set(sources
    ${client_sources}
    ${server_sources})

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
add_hpx_component(itn 
  MODULE bhpx
  SOURCES ${sources} 
  HEADERS ${headers})

