# Copyright (c) 2011 Bryce Lelbach
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(HPX_ADDPSEUDOTARGET_LOADED TRUE)

macro(add_hpx_pseudo_target)
  # Windows is evil
  hpx_info("add_hpx_pseudo_target" "adding pseudo target: ${ARGV}")
  if(NOT WIN32)
    add_custom_target(${ARGV})
  endif()
endmacro()

