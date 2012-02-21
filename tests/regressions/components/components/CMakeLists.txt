# Copyright (c) 2007-2012 Hartmut Kaiser
# Copyright (c)      2011 Bryce Lelbach
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(root "${hpx_SOURCE_DIR}/tests/regressions/components/components")

add_hpx_component(a_undefined_symbol
  HEADER_GLOB "${root}/undefined_symbol.h*"
  SOURCE_GLOB "${root}/undefined_symbol.c*"
  FOLDER "Tests/Regressions/Components"
  AUTOGLOB)

add_hpx_pseudo_dependencies(tests.regressions.components
                            a_undefined_symbol_component)

set_target_properties(a_undefined_symbol_component
    PROPERTIES LINK_FLAGS "-Wl,--allow-shlib-undefined")

