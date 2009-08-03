# Copyright (c) 2007-2009 Dylan Stark
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying 
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

# define required Boost libraries
#set (Boost_COMPONENTS_NEEDED
#     graph)
#find_package(Boost 1.35.0 COMPONENTS ${Boost_COMPONENTS_NEEDED})
#
#INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIR})
#LINK_DIRECTORIES(${Boost_LIBRARY_DIR})

set(ssca_KERNELS
   rmat)

set(ssca_DEPENDENCIES
   graph_component vertex_list_component vertex_component)

foreach(kernel ${ssca_KERNELS})
    set(${kernel}_SOURCES
        ${kernel}.cpp)
    add_hpx_executable(${kernel}
        SOURCES ${${kernel}_SOURCES}
        DEPENDENCIES ${ssca_DEPENDENCIES})
endforeach()
