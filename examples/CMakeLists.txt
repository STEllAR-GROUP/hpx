# Copyright (c) 2007-2008 Hartmut Kaiser
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying 
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set (resolver_client_SOURCES naming/resolver_client.cpp)
add_executable (resolver_client ${resolver_client_SOURCES})
target_link_libraries(resolver_client hpx)

set (resolver_server_SOURCES naming/resolver_server.cpp)
add_executable (resolver_server ${resolver_server_SOURCES})
target_link_libraries(resolver_server hpx)

set (parcelset_client_SOURCES parcelset/parcelset_client.cpp)
add_executable (parcelset_client ${parcelset_client_SOURCES})
target_link_libraries(parcelset_client hpx)

set (parcelset_server_SOURCES parcelset/parcelset_server.cpp)
add_executable (parcelset_server ${parcelset_server_SOURCES})
target_link_libraries(parcelset_server hpx)

