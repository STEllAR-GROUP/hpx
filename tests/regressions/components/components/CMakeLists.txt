# Copyright (c) 2007-2012 Hartmut Kaiser
# Copyright (c)      2011 Bryce Lelbach
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

# Windows requires all symbols to be resolved at link time, this test does not
# make sense on this platform.
if(NOT MSVC)
  add_hpx_component(undefined_symbol
    HEADER_GLOB "undefined_symbol.h*"
    SOURCE_GLOB "undefined_symbol.c*"
    FOLDER "Tests/Regressions/Components"
    AUTOGLOB)

  add_hpx_pseudo_dependencies(tests.regressions.components
                              undefined_symbol_component)

  set_target_properties(undefined_symbol_component
      PROPERTIES LINK_FLAGS "-Wl,--allow-shlib-undefined")
endif()

