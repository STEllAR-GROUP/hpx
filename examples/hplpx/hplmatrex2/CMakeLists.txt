# Distributed under the Boost Software License, Version 1.0. (See accompanying 
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set (HPLMatreX2_HEADERS
    hplmatrex2.hpp
    server/hplmatrex2.hpp
    stubs/hplmatrex2.hpp)

set (HPLMatreX2_SOURCES
    hplmatrex2.cpp)

add_hpx_component(HPLMatreX2
    MODULE hplmatrex2
    SOURCES ${HPLMatreX2_SOURCES}
    HEADERS ${HPLMatreX2_HEADERS})


