# Copyright (c) 2011 Bryce Lelbach
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(HPXLocal_ADDPSEUDOTARGET_LOADED TRUE)

function(hpx_local_add_pseudo_target)
  hpx_local_debug("hpx_local_add_pseudo_target" "adding pseudo target: ${ARGV}")
  if(HPXLocal_WITH_PSEUDO_DEPENDENCIES)
    set(shortened_args)
    foreach(arg ${ARGV})
      hpx_local_shorten_pseudo_target(${arg} shortened_arg)
      set(shortened_args ${shortened_args} ${shortened_arg})
    endforeach()
    hpx_local_debug(
      "hpx_local_add_pseudo_target"
      "adding shortened pseudo target: ${shortened_args}"
    )
    foreach(target ${shortened_args})
      if(NOT TARGET ${target})
        add_custom_target(${shortened_args})
      endif()
    endforeach()
  endif()
endfunction()
