# Distributed under the Boost Software License, Version 1.0. (See accompanying 
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set (HPLMatrex_HEADERS
    hplmatrex.hpp
    server/hplmatrex.hpp
    stubs/hplmatrex.hpp)

set (HPLMatrex_SOURCES
    hplmatrex.cpp)

add_hpx_component(HPLMatrex 
    SOURCES ${HPLMatrex_SOURCES} 
    HEADERS ${HPLMatrex_HEADERS})

