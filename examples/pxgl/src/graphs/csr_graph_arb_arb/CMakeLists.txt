# Copyright (c) 2010-2011 Dylan Stark
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying 
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

################################################################################
# CSR graph representation
set(csr_graph_arb_arb_HEADERS
  ${pxgl_SOURCE_DIR}/pxgl/graphs/csr_graph.hpp
  ${pxgl_SOURCE_DIR}/pxgl/graphs/edge_tuple.hpp
  ${pxgl_SOURCE_DIR}/pxgl/util/scoped_use.hpp
)

set(csr_graph_arb_arb_SOURCES
  csr_graph_arb_arb.cpp
)

set(csr_graph_arb_arb_INI
  csr_graph_arb_arb.ini
)

add_hpx_component(csr_graph_arb_arb
  SOURCES ${csr_graph_arb_arb_SOURCES}
  HEADERS ${csr_graph_arb_arb_HEADERS}
)

