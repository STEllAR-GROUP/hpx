# Copyright (c) 2007-2011 Hartmut Kaiser
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

###############################################################################
set(server_sources
    startup_shutdown.cpp 
    server/startup_shutdown.cpp)

set(server_headers
    server/startup_shutdown.hpp)

###############################################################################
source_group("Source Files" FILES startup_shutdown.cpp)
source_group("Source Files\\Server" FILES server/startup_shutdown.cpp)
source_group("Header Files\\Server" FILES ${server_headers})

###############################################################################
add_hpx_component(startup_shutdown
  SOURCES ${server_sources}
  HEADERS ${server_headers})

