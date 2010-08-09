# Distributed under the Boost Software License, Version 1.0. (See accompanying 
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

# recurse into all components subdirectories and execute the scripts there
set(lud_SUDIRS 
    hplmatrex
    hplmatrex2
    #hplmatrex3
   )

foreach(subdir ${lud_SUDIRS})
    add_subdirectory(${subdir})
endforeach()

# define build targets for this directory
add_hpx_executable(hplpx SOURCES hplpx.cpp DEPENDENCIES HPLMatrex_component)
add_hpx_executable(hplpx2 SOURCES hplpx2.cpp DEPENDENCIES HPLMatrex2_component)
#add_hpx_executable(hplpx3 SOURCES hplpx3.cpp DEPENDENCIES HPLMatrex3_component)
