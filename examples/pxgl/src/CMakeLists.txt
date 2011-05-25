# Copyright (c) 2010-2011 Dylan Stark
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying 
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(src_SUBDIRS
  graphs
  xua
  lcos
)

foreach(subdir ${src_SUBDIRS})
  add_subdirectory(${subdir})
endforeach()

