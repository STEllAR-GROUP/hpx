# Copyright (c) 2010-2011 Dylan Stark
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying 
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(pxgl_lcos_SUBDIRS
  size_have_max
  double_have_max
)

foreach(subdir ${pxgl_lcos_SUBDIRS})
  add_subdirectory(${subdir})
endforeach()

