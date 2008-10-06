# Copyright (c) 2007-2008 Hartmut Kaiser
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying 
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

# Performance measurment helpers
set(performance_TESTS 
    measure_bind measure_getprefix measure_getrange measure_queryid
    measure_registerid measure_resolve measure_unbind measure_unregister
    measure_thread_execution)

foreach(measure ${performance_TESTS})
    set(${measure}_SOURCES ${measure}.cpp high_resolution_timer.hpp)
    add_executable(${measure}_exe ${${measure}_SOURCES})
    set_target_properties(${measure}_exe PROPERTIES OUTPUT_NAME ${measure})
    target_link_libraries(${measure}_exe ${hpx_LIBRARIES} ${Boost_LIBRARIES})
    install(TARGETS ${measure}_exe
        RUNTIME DESTINATION bin
        PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE
                    GROUP_READ GROUP_EXECUTE
                    WORLD_READ WORLD_EXECUTE)
endforeach()

