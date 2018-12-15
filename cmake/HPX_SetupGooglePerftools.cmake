# Copyright (c) 2018 Christopher Hinz
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

if(HPX_WITH_GOOGLE_PERFTOOLS)
    find_package(GooglePerftools)
    if(NOT GOOGLE_PERFTOOLS_FOUND)
        hpx_error("Google Perftools could not be found and HPX_WITH_GOOGLE_PERFTOOLS=On, please specify GOOGLE_PERFTOOLS to point to the root of your Google Perftools installation")
    endif()

    add_library(hpx::gperftools INTERFACE IMPORTED)
    set_property(TARGET hpx::gperftool PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${GOOGLE_PERFTOOLS_INCLUDE_DIR})
    set_property(TARGET hpx::gperftool PROPERTY INTERFACE_LINK_LIBRARIES ${GOOGLE_PERFTOOLS_LIBRARIES})
endif()
