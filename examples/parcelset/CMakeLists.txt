# Copyright (c) 2007-2009 Hartmut Kaiser
# Copyright (c) 2011      Bryce Lelbach
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying 
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

# add pseudo-targets
add_custom_target(examples.parcelset.parcelset_client)
add_custom_target(examples.parcelset.parcelset_server)

# parserset examples (parcelset_server/parcelset_client)
add_hpx_executable(parcelset_client SOURCES parcelset_client.cpp)
add_hpx_executable(parcelset_server SOURCES parcelset_server.cpp)

# add pseudo-target dependencies
add_dependencies(examples.parcelset.parcelset_client parcelset_client)
add_dependencies(examples.parcelset.parcelset_server parcelset_server)

