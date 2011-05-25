# Copyright (c) 2010-2011 Dylan Stark
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying 
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

################################################################################
# CSR graph representation
set(vmap_HEADERS
  ${pxgl_SOURCE_DIR}/pxgl/xua/vmap.hpp
  ${pxgl_SOURCE_DIR}/pxgl/util/scoped_use.hpp
)

set(vmap_SOURCES
  vmap.cpp
)

set(vmap_INI
  vmap.ini
)

add_hpx_component(vmap
  SOURCES ${vmap_SOURCES}
  HEADERS ${vmap_HEADERS}
  INI ${vmap_INI}
  ESSENTIAL
)

