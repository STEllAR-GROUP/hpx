# Copyright (c) 2010-2011 Dylan Stark
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying 
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(have_max_HEADERS
  ${pxgl_SOURCE_DIR}/pxgl/lcos/have_max.hpp
  ${pxgl_SOURCE_DIR}/pxgl/util/component.hpp
)

set(have_max_SOURCES
  size_have_max.cpp
)

set(have_max_INI
  size_have_max.ini
)

add_hpx_component(size_have_max
  SOURCES ${have_max_SOURCES}
  HEADERS ${have_max_HEADERS}
)

