# Copyright (c) 2007-2008 Hartmut Kaiser
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying 
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set (distributing_factory_HEADERS
    distributing_factory.hpp
    stubs/distributing_factory.hpp
    server/distributing_factory.hpp)

set (distributing_factory_SOURCES
    server/distributing_factory.cpp
    distributing_factory.cpp)

add_hpx_component(distributing_factory 
    SOURCES ${distributing_factory_SOURCES} 
    HEADERS ${distributing_factory_HEADERS})

