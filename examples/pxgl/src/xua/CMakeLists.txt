# Copyright (c) 2010-2011 Dylan Stark
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying 
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

# Note: we can only define one component per CMake file
set(pxgl_xua_SUBDIRS
  edge_tuple_vector_constant_range
  edge_tuple_vector_arbitrary_range
  id_vector_arbitrary_range
  double_vector_arbitrary_range
  size_vector_arbitrary_range
  vmap
)

foreach(subdir ${pxgl_xua_SUBDIRS})
  add_subdirectory(${subdir})
endforeach()

