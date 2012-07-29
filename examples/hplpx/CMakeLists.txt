# Copyright (C) 2011 Dan Kogler
# Copyright (C) 2011 Bryce Lelbach
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

add_definitions(-DHPX_ACTION_ARGUMENT_LIMIT=7)
add_definitions(-DHPX_FUNCTION_ARGUMENT_LIMIT=10)

add_subdirectory(hplmatrex)
add_subdirectory(smphplmatrex)

set(hplpx_DEPENDENCIES
    hplmatrex_component
    lublock_component)

add_hpx_executable(hplpx
  SOURCES hplpx.cpp
  DEPENDENCIES ${hplpx_DEPENDENCIES}
  FOLDER "Examples/Applications/HplPx")

add_hpx_executable(smphplpx
  SOURCES smphplpx.cpp
  DEPENDENCIES smphplmatrex_component
  FOLDER "Examples/Applications/HplPx")

add_hpx_pseudo_dependencies(examples.hplpx hplpx_exe)
add_hpx_pseudo_dependencies(examples.hplpx smphplpx_exe)
