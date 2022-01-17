# Copyright (c) 2007-2009 Hartmut Kaiser
# Copyright (c) 2011 Matt Anderson
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

# ##############################################################################
set(client_sources random_mem_access.cpp)

# ##############################################################################
set(client_headers random_mem_access.hpp)

set(server_headers server/random_mem_access.hpp)

# ##############################################################################
set(headers ${client_headers} ${server_headers})

# ##############################################################################
source_group("Source Files" FILES ${client_sources})
source_group("Header Files" FILES ${client_headers})
source_group("Header Files\\Server" FILES ${server_headers})

# ##############################################################################
add_hpx_component(
  random_mem_access INTERNAL_FLAGS NOEXPORT MODULE random_mem_access
  SOURCES ${client_sources}
  HEADERS ${headers}
  FOLDER "Examples/RandomMemoryAccess/random_mem_access"
)
