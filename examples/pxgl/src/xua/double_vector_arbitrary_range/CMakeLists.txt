# Copyright (c) 2010-2011 Dylan Stark
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying 
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

################################################################################
# Vector representation
set(numeric_HEADERS
  ${pxgl_SOURCE_DIR}/pxgl/xua/range.hpp
  ${pxgl_SOURCE_DIR}/pxgl/xua/arbitrary_distribution.hpp
  ${pxgl_SOURCE_DIR}/pxgl/xua/numeric.hpp
  ${pxgl_SOURCE_DIR}/pxgl/util/scoped_use.hpp
  ${pxgl_SOURCE_DIR}/pxgl/util/component.hpp
)

set(numeric_SOURCES
  double_vector_arbitrary_range.cpp
)

set(numeric_INI
  double_vector_arbitrary_range.ini
)

add_hpx_component(double_numeric_arbitrary_range
  SOURCES ${numeric_SOURCES}
  HEADERS ${numeric_HEADERS}
)

