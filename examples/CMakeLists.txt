# Copyright (c) 2007-2008 Hartmut Kaiser
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying 
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

# dgas examples (resolver_server/resolver_client)
set(resolver_client_SOURCES naming/resolver_client.cpp)
add_executable(resolver_client ${resolver_client_SOURCES})
target_link_libraries(resolver_client ${hpx_LIBRARIES} ${Boost_LIBRARIES})

set(resolver_server_SOURCES naming/resolver_server.cpp)
add_executable(resolver_server ${resolver_server_SOURCES})
target_link_libraries(resolver_server ${hpx_LIBRARIES} ${Boost_LIBRARIES})


# parserset examples (parcelset_server/parcelset_client)
set(parcelset_client_SOURCES parcelset/parcelset_client.cpp)
add_executable(parcelset_client ${parcelset_client_SOURCES})
target_link_libraries(parcelset_client ${hpx_LIBRARIES} ${Boost_LIBRARIES})

set(parcelset_server_SOURCES parcelset/parcelset_server.cpp)
add_executable(parcelset_server ${parcelset_server_SOURCES})
target_link_libraries(parcelset_server ${hpx_LIBRARIES} ${Boost_LIBRARIES})


# accumulator_client example
set(accumulator_SOURCES accumulator/accumulator_client.cpp)
add_executable(accumulator_client ${accumulator_SOURCES})
target_link_libraries(
    accumulator_client                    # executable
    accumulator                           # components it depends on
    ${hpx_LIBRARIES} ${Boost_LIBRARIES}   # libraries it dependson
)


# simple_accumulator_client example
set(simple_accumulator_SOURCES accumulator/simple_accumulator_client.cpp)
add_executable(simple_accumulator_client ${simple_accumulator_SOURCES})
target_link_libraries(
    simple_accumulator_client             # executable
    simple_accumulator                    # components it depends on
    ${hpx_LIBRARIES} ${Boost_LIBRARIES}   # libraries it dependson
)

