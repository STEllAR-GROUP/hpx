# Copyright (c) 2007-2009 Hartmut Kaiser
# Copyright (c)      2011 Bryce Adelstein-Lelbach
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

if(NOT HPX_WITH_DISTRIBUTED_RUNTIME)
  return()
endif()

set(components simple_central_tuplespace)

foreach(component ${components})

  add_hpx_component(
    ${component} INTERNAL_FLAGS NOEXPORT AUTOGLOB
    HEADER_GLOB "${component}.h*"
    SOURCE_GLOB "${component}.c*"
    FOLDER "Examples/TupleSpace/${component}"
  )

endforeach()
