set (tlf_HEADERS
    tlf.hpp
    server/tlf.hpp
    stubs/tlf.hpp)

set (tlf_SOURCES
    tlf.cpp)

add_hpx_component(tlf
    SOURCES ${tlf_SOURCES} 
    HEADERS ${tlf_HEADERS})
