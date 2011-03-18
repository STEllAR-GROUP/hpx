# Copyright (c) 2011 Bryce Lelbach
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying 
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

add_hpx_library_sources(hpx
  address.cpp
  bulk_resolver_client.cpp
  full_address.cpp
  locality.cpp
  name.cpp
  resolver_client.cpp
  resolver_server.cpp)

add_subdirectory(detail)
add_subdirectory(server)

