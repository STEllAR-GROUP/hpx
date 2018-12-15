# Copyright (c) 2018 Christopher Hinz
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

if(HPX_WITH_APEX)
    add_library(hpx::apex INTERFACE IMPORTED)

    # Workaround: The already existing apex target does not set its own include directory.
    set_property(TARGET hpx::apex PROPERTY INTERFACE_INCLUDE_DIRECTORIES $<BUILD_INTERFACE:${APEX_SOURCE_DIR}>
                                                                         $<INSTALL_INTERFACE:include>)
    set_property(TARGET hpx::apex PROPERTY INTERFACE_LINK_LIBRARIES apex)

    if(APEX_WITH_MSR)
        add_library(hpx::msr INTERFACE IMPORTED)
        # TODO: Use an absolute path.
        set_property(TARGET hpx::msr PROPERTY INTERFACE_LINK_LIBRARIES -L${MSR_ROOT}/lib -lmsr)

        set_property(TARGET hpx::apex PROPERTY INTERFACE_LINK_LIBRARIES hpx::msr)
    endif()
    if(APEX_WITH_ACTIVEHARMONY)
        add_library(hpx::activeharmony INTERFACE IMPORTED)
        # TODO: Use an absolute path.
        set_property(TARGET hpx::activeharmony PROPERTY INTERFACE_LINK_LIBRARIES -L${ACTIVEHARMONY_ROOT}/lib -lharmony)

        set_property(TARGET hpx::apex PROPERTY INTERFACE_LINK_LIBRARIES hpx::activeharmony)
    endif()
    if(APEX_WITH_OTF2)
        add_library(hpx::otf2 INTERFACE IMPORTED)
        # TODO: Use an absolute path.
        set_property(TARGET hpx::otf2 PROPERTY INTERFACE_LINK_LIBRARIES -L${OTF2_ROOT}/lib -lotf2)

        set_property(TARGET hpx::apex PROPERTY INTERFACE_LINK_LIBRARIES hpx::otf2)
    endif()

    # handle optional ITTNotify library
    if(HPX_WITH_ITTNOTIFY)
        add_subdirectory(apex/src/ITTNotify)
        if(NOT ITTNOTIFY_FOUND)
            hpx_error("ITTNotify could not be found and HPX_WITH_ITTNOTIFY=On")
        endif()

        add_library(hpx::ittnotify INTERFACE IMPORTED)
        set_property(TARGET hpx::ittnotify PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${ITTNOTIFY_SOURCE_DIR})

        hpx_add_config_define(HPX_HAVE_ITTNOTIFY 1)

        set_property(TARGET hpx::apex PROPERTY INTERFACE_LINK_LIBRARIES hpx::ittnotify)
    endif()
endif()
