# Copyright (c) 2010-2011 Dylan Stark
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying 
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

################################################################################
set(bc_sssp_HEADERS
  ${pxgl_SOURCE_DIR}/pxgl/xua/arbitrary_distribution.hpp
  ${pxgl_SOURCE_DIR}/pxgl/xua/range.hpp
  ${pxgl_SOURCE_DIR}/pxgl/xua/control.hpp
  ${pxgl_SOURCE_DIR}/pxgl/xua/vector.hpp
  ${pxgl_SOURCE_DIR}/pxgl/xua/numeric.hpp
  ${pxgl_SOURCE_DIR}/pxgl/graphs/csr_graph.hpp
  ${pxgl_SOURCE_DIR}/pxgl/graphs/edge_tuple.hpp
  bc_sssp.hpp
)

set(bc_sssp_SOURCES
  bc_sssp.cpp
)

set(bc_sssp_INI
  bc_sssp.ini
)

add_hpx_component(bc_sssp
  SOURCES ${bc_sssp_SOURCES}
  HEADERS ${bc_sssp_HEADERS}
  INI ${bc_sssp_INI}
  ESSENTIAL
)

