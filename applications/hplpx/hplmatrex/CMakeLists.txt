# Distributed under the Boost Software License, Version 1.0. (See accompanying 
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set (HPLMatreX_HEADERS
    hplmatrex.hpp
    server/hplmatrex.hpp
    stubs/hplmatrex.hpp)

set (HPLMatreX_SOURCES
    hplmatrex.cpp)

add_hpx_component(HPLMatreX 
    SOURCES ${HPLMatreX_SOURCES} 
    HEADERS ${HPLMatreX_HEADERS})

