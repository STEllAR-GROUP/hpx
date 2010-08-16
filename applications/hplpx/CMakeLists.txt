# Distributed under the Boost Software License, Version 1.0. (See accompanying 
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

# recurse into all components subdirectories and execute the scripts there
set(lud_SUDIRS
    #hplmatrex
    #hplmatrex2
    hplmatrex3
#    hplmatrex4
   )

foreach(subdir ${lud_SUDIRS})
    add_subdirectory(${subdir})
endforeach()

# define build targets for this directory
#add_hpx_executable(hplpx SOURCES hplpx.cpp DEPENDENCIES HPLMatreX_component)
#add_hpx_executable(hplpx2 SOURCES hplpx2.cpp DEPENDENCIES HPLMatreX2_component)
add_hpx_executable(hplpx3 SOURCES hplpx3.cpp DEPENDENCIES HPLMatreX3_component)
#add_hpx_executable(hplpx4 SOURCES hplpx4.cpp DEPENDENCIES HPLMatreX4_component)
