if(HPX_WITH_VALGRIND)
    find_package(Valgrind)
    if(NOT VALGRIND_FOUND)
        hpx_error("Valgrind could not be found and HPX_WITH_VALGRIND=On, please specify VALGRIND_ROOT to point to the root of your Valgrind installation")
    endif()

    add_library(hpx::valgrind INTERFACE IMPORTED)
    set_property(TARGET hpx::valgrind PROPERTY INTERFACE_SYSTEM_INCLUDE_DIRECTORIES ${VALGRIND_INCLUDE_DIR})

    hpx_add_config_define(HPX_HAVE_VALGRIND)
endif()