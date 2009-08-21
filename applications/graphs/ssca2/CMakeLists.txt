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

set(ssca_SUDIRS 
    ssca2)
    
foreach(subdir ${ssca_SUDIRS})
    add_subdirectory(${subdir})
endforeach()

set(ssca_DEPENDENCIES
   graph_component vertex_list_component vertex_component
   distributed_set_component ssca2_component)

set(ssca2_benchmark_SOURCES
    ssca2_benchmark.cpp)
    
add_hpx_executable(ssca2_benchmark
    SOURCES ${ssca2_benchmark_SOURCES}
    DEPENDENCIES ${ssca_DEPENDENCIES})
