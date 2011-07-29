# Copyright (C) 2011 Daniel Kogler
# Copyright (C) 2011 Bryce Lelbach
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

add_subdirectory(barnes_hut)

set(barnes_hut_DEPENDENCIES
    bhcontroller_component
    bhnode_component)

add_hpx_executable(barnes
  MODULE bhcontroller
  SOURCES barnes_hut.cpp
  DEPENDENCIES ${barnes_hut_DEPENDENCIES})

add_hpx_pseudo_dependencies(examples.barnes_hut barnes_exe)
