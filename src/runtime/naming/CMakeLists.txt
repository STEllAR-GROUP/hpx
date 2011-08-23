# Copyright (c) 2011 Bryce Lelbach
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying 
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

add_hpx_library_sources(hpx
  address.cpp
  locality.cpp
  name.cpp
  resolver_server.cpp)

if("${HPX_AGAS_VERSION}" STREQUAL "1")
  add_hpx_library_sources(hpx
    resolver_client.cpp
    bulk_resolver_client.cpp)

  add_subdirectory(detail)
endif()

add_subdirectory(server)

