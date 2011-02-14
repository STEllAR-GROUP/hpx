message("-- Configuring HPX")

find_path(HPX_INCDIR "hpx/hpx.hpp" ${HPX_ROOT}/include
    /usr/local/hpx/include /usr/local/include /usr/include)
include_directories(${HPX_INCDIR})

foreach(HPX_LIB hpx hpx_serialization hpx_component_distributing_factory)
    find_library("${HPX_LIB}_lib" ${HPX_LIB}
        ${HPX_ROOT}/lib /usr/local/hpx/lib /usr/local/lib /usr/lib /lib)
    link_libraries(${${HPX_LIB}_lib})
    if(NOT "${HPX_LIB}_lib")
        message("   *** Could not locate the ${HPX_LIB} library *** ")
    endif()
endforeach(HPX_LIB)
