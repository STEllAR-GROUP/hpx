# Copyright (c) 2007-2009 Hartmut Kaiser
# Copyright (c)      2011 Bryce Adelstein-Lelbach
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(components
    managed_accumulator
    simple_accumulator
   )

foreach(component ${components})

  add_hpx_component(${component}
    AUTOGLOB
    HEADER_GLOB "${component}.h*"
    SOURCE_GLOB "${component}.c*"
    FOLDER "Examples/Quickstart/Accumulators/${component}")

endforeach()

