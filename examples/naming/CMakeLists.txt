# Copyright (c) 2007-2008 Hartmut Kaiser
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying 
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

# agas examples (resolver_server/resolver_client)
set(resolver_client_SOURCES resolver_client.cpp)
add_executable(resolver_client_exe ${resolver_client_SOURCES})
set_target_properties(resolver_client_exe PROPERTIES OUTPUT_NAME resolver_client)
target_link_libraries(resolver_client_exe ${hpx_LIBRARIES} ${Boost_LIBRARIES})

set(resolver_server_SOURCES resolver_server.cpp)
add_executable(resolver_server_exe ${resolver_server_SOURCES})
set_target_properties(resolver_server_exe PROPERTIES OUTPUT_NAME resolver_server)
target_link_libraries(resolver_server_exe ${hpx_LIBRARIES} ${Boost_LIBRARIES})

