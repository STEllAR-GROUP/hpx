if(STDEXEC_ROOT AND NOT Stdexec_ROOT)
    set(Stdexec_ROOT
        ${STDEXEC_ROOT}
        CACHE PATH "STDEXEC base directory"
    )
    unset(STDEXEC_ROOT CACHE)
endif()

if(HPX_WITH_FETCH_STDEXEC AND NOT Stdexec_ROOT)
    hpx_info(
        "HPX_WITH_FETCH_STDEXEC=${HPX_WITH_FETCH_STDEXEC}, Stdexec will be fetched using CMake's FetchContent."
    )
    if(UNIX)
        include(FetchContent)
        FetchContent_Declare(
            Stdexec
            GIT_REPOSITORY https://github.com/NVIDIA/stdexec.git
            GIT_TAG        main
        )
        FetchContent_MakeAvailable(Stdexec)
    endif()

    add_library(STDEXEC::stdexec INTERFACE IMPORTED)
    target_include_directories(STDEXEC::stdexec INTERFACE ${stdexec_SOURCE_DIR}/include)
    target_link_libraries(STDEXEC::stdexec INTERFACE ${Stdexec_LIBRARY})
elseif(HPX_WITH_STDEXEC)
    find_package(Stdexec REQUIRED)

    if(NOT Stdexec_FOUND)
        hpx_error(
            "Stdexec could not be found, please specify Stdexec_ROOT to point to the correct location"
        )
    endif()
endif()