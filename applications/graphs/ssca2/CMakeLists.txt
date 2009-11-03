# Copyright (c) 2007-2009 Dylan Stark
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying 
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(ssca_SUDIRS 
    ssca2
    props
    pbreak
    reduce_max)
    
foreach(subdir ${ssca_SUDIRS})
    add_subdirectory(${subdir})
endforeach()

set(ssca_DEPENDENCIES
   graph_component vertex_component
   distributed_set_component 
   distributed_map_component
   ssca2_component
   )

set (ssca2_HEADERS
     ssca2_benchmark.hpp
    )
    
set(ssca2_benchmark_SOURCES
    ssca2_benchmark.cpp
    kernel3.cpp
    kernel4.cpp)
    
add_hpx_executable(ssca2_benchmark
    SOURCES ${ssca2_benchmark_SOURCES}
    HEADERS ${ssca2_benchmark_HEADERS}
    DEPENDENCIES ${ssca_DEPENDENCIES})
