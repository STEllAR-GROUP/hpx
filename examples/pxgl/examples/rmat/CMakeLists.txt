# Copyright (c) 2010-2011 Dylan Stark
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying 
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(rmat_DEPENDENCIES
  edge_tuple_vector_arbitrary_range_component
)

set(rmat_HEADERS
  ${pxgl_SOURCE_DIR}/pxgl/xua/arbitrary_distribution.hpp
  ${pxgl_SOURCE_DIR}/pxgl/xua/range.hpp
  ${pxgl_SOURCE_DIR}/pxgl/xua/control.hpp
  ${pxgl_SOURCE_DIR}/pxgl/util/apply.hpp
  ${pxgl_SOURCE_DIR}/pxgl/util/hpx.hpp
  ${pxgl_SOURCE_DIR}/pxgl/graphs/edge_tuple.hpp
)

set(rmat_SOURCES
  rmat.cpp
)

add_hpx_executable(generate_rmat_data
  DEPENDENCIES ${rmat_DEPENDENCIES}
  HEADERS ${rmat_HEADERS}
  SOURCES ${rmat_SOURCES}
)

add_hpx_pseudo_dependencies(examples.pxgl generate_rmat_data_exe)

