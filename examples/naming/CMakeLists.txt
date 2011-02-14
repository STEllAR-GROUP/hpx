# Copyright (c) 2007-2009 Hartmut Kaiser
# Copyright (c) 2011      Bryce Lelbach
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying 
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

# add pseudo-targets
add_custom_target(examples.naming.resolver_client)
add_custom_target(examples.naming.resolver_server)

# agas examples (resolver_server/resolver_client)
add_hpx_executable(resolver_client SOURCES resolver_client.cpp)
add_hpx_executable(resolver_server SOURCES resolver_server.cpp)

# add pseudo-target dependencies
add_dependencies(examples.naming.resolver_client resolver_client_exe)
add_dependencies(examples.naming.resolver_server resolver_server_exe)

