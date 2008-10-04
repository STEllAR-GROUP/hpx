# Copyright (c) 2007-2008 Hartmut Kaiser
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying 
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set (hpx_runtime_SOURCES hpx_runtime.cpp)
add_executable (hpx_runtime_exe ${hpx_runtime_SOURCES})
set_target_properties(hpx_runtime_exe PROPERTIES 
    OUTPUT_NAME hpx_runtime${CMAKE_DEBUG_POSTFIX})
target_link_libraries(hpx_runtime_exe hpx ${Boost_LIBRARIES})

install(TARGETS hpx_runtime_exe
    RUNTIME DESTINATION bin
    PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE
                GROUP_READ GROUP_EXECUTE
                WORLD_READ WORLD_EXECUTE)
