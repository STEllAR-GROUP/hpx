# Copyright (c) 2018 Christopher Hinz
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

if(HPX_WITH_APEX)
    set(_hpx_apex_no_update)
    if(HPX_WITH_APEX_NO_UPDATE)
        set(_hpx_apex_no_update NO_UPDATE)
    endif()
    set(_hpx_apex_tag "v2.1.0")
    if(HPX_WITH_APEX_TAG)
        message("Overriding APEX git tag ${_hpx_apex_tag} with ${HPX_WITH_APEX_TAG}")
        set(_hpx_apex_tag ${HPX_WITH_APEX_TAG})
    endif()

    # We want to track parent dependencies
    hpx_add_config_define(HPX_HAVE_THREAD_PARENT_REFERENCE)
    # handle APEX library
    include(GitExternal)
    git_external(apex
            https://github.com/khuck/xpress-apex.git
            ${_hpx_apex_tag}
            ${_hpx_apex_no_update}
            VERBOSE)

    LIST(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/apex/cmake/Modules")
    add_subdirectory(apex/src/apex)

    if(NOT APEX_FOUND)
        hpx_error("Apex could not be found and HPX_WITH_APEX=On")
    endif()
    if(AMPLIFIER_FOUND)
        hpx_error("AMPLIFIER_FOUND has been set. Please disable the use of the Intel Amplifier (WITH_AMPLIFIER=Off) in order to use Apex")
    endif()
endif()
