# Copyright (c) 2007-2008 Hartmut Kaiser
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying 
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set (console_HEADERS
    console.hpp
    server/console_logging.hpp)

set (console_SOURCES
    console.cpp
    server/console_logging.cpp)

set(console_INI
    console.ini)

add_hpx_component(console 
    SOURCES ${console_SOURCES} 
    HEADERS ${console_HEADERS}
    INI ${console_INI})

