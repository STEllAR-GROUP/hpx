
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

# recurse into all components subdirectories and execute the scripts there

add_subdirectory(hplmatrex)
add_subdirectory(hplmatrex2)


# define build targets for this directory
add_hpx_executable(hplpx1
        MODULE hplmatrex
        SOURCES hplpx.cpp
        DEPENDENCIES HPLMatreX_component
        ESSENTIAL)
add_hpx_executable(hplpx2
        MODULE hplmatrex2
        SOURCES hplpx2.cpp
        DEPENDENCIES HPLMatreX2_component
        ESSENTIAL)

add_custom_target(applications.hplpx.hplmatrex)
add_dependencies(applications.hplpx hplpx.hplmatrex)
add_custom_target(applications.hplpx.hplmatrex2)
add_dependencies(applications.hplpx hplpx2.hplmatrex2)

