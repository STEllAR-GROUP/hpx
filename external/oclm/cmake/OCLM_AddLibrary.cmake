
macro(oclm_add_library LIB)
    if(OCLM_LIBRARIES)
        set(OCLM_LIBRARIES ${OCLM_LIBRARIES} ${LIB})
    else()
        set(OCLM_LIBRARIES ${LIB})
    endif()
endmacro()
