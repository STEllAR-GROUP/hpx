# Copyright (c) 2010-2011 Dylan Stark
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying 
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

################################################################################
# CSR graph representation
set(dynamic_graph_HEADERS
  ${pxgl_SOURCE_DIR}/pxgl/graphs/dynamic_graph.hpp
  ${pxgl_SOURCE_DIR}/pxgl/graphs/edge_tuple.hpp
  ${pxgl_SOURCE_DIR}/pxgl/util/scoped_use.hpp
)

set(dynamic_graph_SOURCES
  dynamic_graph.cpp
)

set(dynamic_graph_INI
  dynamic_graph.ini
)

add_hpx_component(dynamic_graph
  SOURCES ${dynamic_graph_SOURCES}
  HEADERS ${dynamic_graph_HEADERS}
  INI ${dynamic_graph_INI}
  ESSENTIAL
)

