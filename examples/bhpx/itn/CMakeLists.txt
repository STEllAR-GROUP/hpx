set (itn_HEADERS
    itn.hpp
    server/itn.hpp
    stubs/itn.hpp)

set (itn_SOURCES
    server/itn.cpp
    itn.cpp)

add_hpx_component(itn
    SOURCES ${itn_SOURCES} 
    HEADERS ${itn_HEADERS})