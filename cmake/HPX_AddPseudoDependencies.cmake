# Copyright (c) 2011 Bryce Lelbach
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(HPX_ADDPSEUDODEPENDENCIES_LOADED TRUE)

macro(add_hpx_pseudo_dependencies)
  # Windows is evil
  if(NOT MSVC AND NOT MINGW)
    add_dependencies(${ARGV})
  endif()
endmacro()

