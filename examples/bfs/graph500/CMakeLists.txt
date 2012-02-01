# Copyright (c) 2011 Matt Anderson
# Copyright (c) 2007-2012 Hartmut Kaiser
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(client_sources
    point_module.cpp)

set(client_headers
    point.hpp)

set(stubs_headers
    stubs/point.hpp)

set(server_headers
    server/point.hpp
    server/graph_generator.hpp
    server/mod_arith_64bit.hpp
    server/splittable_mrg.hpp
    server/mrg_transitions.hpp)

set(server_sources
    server/point.cpp
    server/graph_generator.cpp
    server/splittable_mrg.cpp)

set(sources
    ${client_sources}
    ${server_sources})

set(headers
    ${common_headers}
    ${client_headers}
    ${stubs_headers}
    ${server_headers})

add_hpx_component(graph500
  MODULE graph500
  SOURCES ${sources}
  HEADERS ${headers}
  DEPENDENCIES iostreams_component
  FOLDER "Examples/Bfs")

