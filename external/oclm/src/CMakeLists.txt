
#  Copyright (c) 2012 Thomas Heller
#
#  Distributed under the Boost Software License, Version 1.0. (See accompanying
#  file LICENSE_1_0.txt or copy at http:#www.boost.org/LICENSE_1_0.txt)

set(
    OCLM_LIBRARY_SOURCES
    context.cpp
    context_info.cpp
    command_queue.cpp
    device.cpp
    device_info.cpp
    platform.cpp
    platform_info.cpp
    program.cpp
)

add_library(oclm SHARED ${OCLM_LIBRARY_SOURCES})
target_link_libraries(oclm ${OCLM_LIBRARIES})
