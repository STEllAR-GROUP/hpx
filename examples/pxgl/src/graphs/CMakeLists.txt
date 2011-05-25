# Copyright (c) 2010-2011 Dylan Stark
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying 
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

# Note: we can only define one component per CMake file
set(pxgl_graphs_SUBDIRS
  csr_graph_arb_const
  csr_graph_arb_arb
  dynamic_graph
)

foreach(subdir ${pxgl_graphs_SUBDIRS})
  add_subdirectory(${subdir})
endforeach()

