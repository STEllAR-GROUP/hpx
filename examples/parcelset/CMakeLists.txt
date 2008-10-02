# Copyright (c) 2007-2008 Hartmut Kaiser
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying 
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

# parserset examples (parcelset_server/parcelset_client)
set(parcelset_client_SOURCES parcelset_client.cpp)
add_executable(parcelset_client_exe ${parcelset_client_SOURCES})
set_target_properties(parcelset_client_exe PROPERTIES OUTPUT_NAME parcelset_client)
target_link_libraries(parcelset_client_exe ${hpx_LIBRARIES} ${Boost_LIBRARIES})

set(parcelset_server_SOURCES parcelset_server.cpp)
add_executable(parcelset_server_exe ${parcelset_server_SOURCES})
set_target_properties(parcelset_server_exe PROPERTIES OUTPUT_NAME parcelset_server)
target_link_libraries(parcelset_server_exe ${hpx_LIBRARIES} ${Boost_LIBRARIES})


