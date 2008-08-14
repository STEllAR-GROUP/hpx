# Copyright (c) 2007-2008 Hartmut Kaiser
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying 
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set (hpx_runtime_SOURCES hpx_runtime.cpp)
add_executable (hpx_runtime ${hpx_runtime_SOURCES})
target_link_libraries(hpx_runtime hpx ${Boost_LIBRARIES})

set (hpx_runtime_pure_SOURCES hpx_runtime_pure.cpp)
add_executable (hpx_runtime_pure ${hpx_runtime_pure_SOURCES})
target_link_libraries(hpx_runtime_pure hpx ${Boost_LIBRARIES})
