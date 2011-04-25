# Copyright (C) 2011 Dan Kogler
# Copyright (C) 2011 Bryce Lelbach
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

add_subdirectory(hplmatrex)
add_subdirectory(hplmatrex2)

add_hpx_executable(hplpx1
        MODULE hplmatrex
        SOURCES hplpx.cpp
        DEPENDENCIES hplmatrex_component)

add_hpx_pseudo_target(examples.hplpx.hplmatrex)
add_hpx_pseudo_dependencies(examples.hplpx examples.hplpx.hplmatrex)
add_hpx_pseudo_dependencies(examples.hplpx.hplmatrex hplpx1_exe)

add_hpx_executable(hplpx2
        MODULE hplmatrex2
        SOURCES hplpx2.cpp
        DEPENDENCIES hplmatrex2_component)

add_hpx_pseudo_target(examples.hplpx.hplmatrex2)
add_hpx_pseudo_dependencies(examples.hplpx examples.hplpx.hplmatrex2)
add_hpx_pseudo_dependencies(examples.hplpx.hplmatrex2 hplpx2_exe)

