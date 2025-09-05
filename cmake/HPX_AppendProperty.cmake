# Copyright (c) 2011-2012 Bryce Lelbach
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

function(hpx_append_property target property)
  set_property(
    TARGET ${target}
    PROPERTY ${property} "${ARGN}"
    APPEND
  )
endfunction()
