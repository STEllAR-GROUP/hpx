# Copyright (c) 2010-2011 Dylan Stark
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying 
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

################################################################################
# Vector representation
set(vector_HEADERS
  ${pxgl_SOURCE_DIR}/pxgl/xua/range.hpp
  ${pxgl_SOURCE_DIR}/pxgl/xua/arbitrary_distribution.hpp
  ${pxgl_SOURCE_DIR}/pxgl/xua/vector.hpp
  ${pxgl_SOURCE_DIR}/pxgl/util/scoped_use.hpp
  ${pxgl_SOURCE_DIR}/pxgl/util/component.hpp
)

set(vector_SOURCES
  id_vector_arbitrary_range.cpp
)

set(vector_INI
  id_vector_arbitrary_range.ini
)

add_hpx_component(id_vector_arbitrary_range
  SOURCES ${vector_SOURCES}
  HEADERS ${vector_HEADERS}
)

