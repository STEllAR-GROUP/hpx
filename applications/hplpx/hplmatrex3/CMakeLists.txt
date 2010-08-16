# Distributed under the Boost Software License, Version 1.0. (See accompanying 
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set (HPLMatrex3_HEADERS
    hplmatrex3.hpp
    server/hplmatrex3.hpp
    stubs/hplmatrex3.hpp)

set (HPLMatrex3_SOURCES
    hplmatrex3.cpp)

add_hpx_component(HPLMatreX3 
    SOURCES ${HPLMatrex3_SOURCES} 
    HEADERS ${HPLMatrex3_HEADERS})

