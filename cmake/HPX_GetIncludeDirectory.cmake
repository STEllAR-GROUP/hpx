# Copyright (c) 2011 Bryce Lelbach
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(HPX_GETINCLUDEDIRECTORY_LOADED TRUE)

macro(hpx_get_include_directory variable)
  set(dir "")
  if(hpx_SOURCE_DIR)
    set(dir "-I${hpx_SOURCE_DIR}")
  elseif(HPX_ROOT)
    set(dir "-I${HPX_ROOT}/include")
  elseif($ENV{HPX_ROOT})
    set(dir "-I$ENV{HPX_ROOT}/include")
  endif()

  set(${variable} "${dir}")
endmacro()

