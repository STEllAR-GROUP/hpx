# Copyright (c) 2007-2009 Dylan Stark
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying 
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(rmat_SUDIRS 
   ) 
    
foreach(subdir ${rmat_SUDIRS})
    add_subdirectory(${subdir})
endforeach()

set(rmat_DEPENDENCIES
   graph_component
   )

set(rmat_HEADERS
   )
    
set(rmat_SOURCES
   rmat.cpp
   )
    
add_hpx_executable(rmat
    SOURCES ${rmat_SOURCES}
    HEADERS ${rmat_HEADERS}
    DEPENDENCIES ${rmat_DEPENDENCIES})
