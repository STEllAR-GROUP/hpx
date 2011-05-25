# Copyright (c) 2010-2011 Dylan Stark
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying 
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

################################################################################
set(sgab_SUBDIRS
  bc_sssp
)

foreach(subdir ${sgab_SUBDIRS})
  add_subdirectory(${subdir})
endforeach()

################################################################################
set(sgab_DEPENDENCIES
  csr_graph_arb_arb_component
  edge_tuple_vector_arbitrary_range_component
  size_have_max_component
  id_vector_arbitrary_range_component
  dynamic_graph_component
  vmap_component
  size_vector_arbitrary_range_component
  double_numeric_arbitrary_range_component
  double_have_max_component
  bc_sssp_component
)

set(sgab_HEADERS
  ${pxgl_SOURCE_DIR}/pxgl/graphs/csr_graph.hpp
  ${pxgl_SOURCE_DIR}/pxgl/graphs/edge_tuple.hpp
  ${pxgl_SOURCE_DIR}/pxgl/xua/arbitrary_distribution.hpp
  ${pxgl_SOURCE_DIR}/pxgl/xua/range.hpp
  ${pxgl_SOURCE_DIR}/pxgl/xua/control.hpp
  ${pxgl_SOURCE_DIR}/pxgl/xua/control_implementations.hpp
  ${pxgl_SOURCE_DIR}/pxgl/xua/vmap_client.hpp
  ${pxgl_SOURCE_DIR}/pxgl/lcos/have_max.hpp
  ${pxgl_SOURCE_DIR}/pxgl/util/apply.hpp
  ${pxgl_SOURCE_DIR}/pxgl/util/hpx.hpp
  ${pxgl_SOURCE_DIR}/examples/sgab/bc_sssp/bc_sssp_client.hpp
)

set(sgab_SOURCES
  sgab.cpp
)

add_hpx_executable(sgab_run
  DEPENDENCIES ${sgab_DEPENDENCIES}
  HEADERS ${sgab_HEADERS}
  SOURCES ${sgab_SOURCES}
  ESSENTIAL
)

