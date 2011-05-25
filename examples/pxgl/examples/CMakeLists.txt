# Copyright (c) 2010-2011 Dylan Stark
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying 
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(examples_SUBDIRS
  sgab
  rmat
)

foreach(subdir ${examples_SUBDIRS})
  add_subdirectory(${subdir})
endforeach()
