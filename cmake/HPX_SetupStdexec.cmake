#  Copyright (c) 2024 Isidoros Tsaousis-Seiras
#
#  SPDX-License-Identifier: BSL-1.0
#  Distributed under the Boost Software License, Version 1.0. (See accompanying
#  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

if(HPX_WITH_STDEXEC AND NOT TARGET STDEXEC::stdexec)
    hpx_add_config_define(HPX_HAVE_STDEXEC)

    if(HPX_WITH_FETCH_STDEXEC)
        hpx_info(
            "HPX_WITH_FETCH_STDEXEC=${HPX_WITH_FETCH_STDEXEC}, Stdexec will be fetched using CMake's FetchContent."
        )
        if(UNIX)
            include(FetchContent)
            message("FETCHING STDEXEC")
            FetchContent_Declare(
                Stdexec
                GIT_REPOSITORY https://github.com/NVIDIA/stdexec.git
                GIT_TAG        ${HPX_WITH_STDEXEC_TAG}
            )
            FetchContent_MakeAvailable(Stdexec)
        endif()

        # add_library(STDEXEC::stdexec INTERFACE IMPORTED)
        # target_include_directories(STDEXEC::stdexec INTERFACE ${stdexec_SOURCE_DIR})
        # target_link_libraries(STDEXEC::stdexec INTERFACE ${Stdexec_LIBRARY})
    else()
        find_package(Stdexec REQUIRED)

        if(NOT Stdexec_FOUND)
            hpx_error(
                "Stdexec could not be found, please specify Stdexec_ROOT to point to the correct location"
            )
        endif()
    endif()
endif()