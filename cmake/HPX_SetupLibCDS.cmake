# Copyright (c) 2007-2019 Hartmut Kaiser
# Copyright (c) 2011-2014 Thomas Heller
# Copyright (c) 2007-2008 Chirag Dekate
# Copyright (c)      2011 Bryce Lelbach
# Copyright (c)      2011 Vinay C Amatya
# Copyright (c)      2013 Jeroen Habraken
# Copyright (c) 2014-2016 Andreas Schaefer
# Copyright (c) 2017      Abhimanyu Rawat
# Copyright (c) 2017      Google
# Copyright (c) 2017      Taeguk Kwon
# Copyright (c) 2018 Christopher Hinz
# Copyright (c) 2020      Weile Wei
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

if(HPX_WITH_LIBCDS AND NOT TARGET LibCDS::cds)

    include(FetchContent)

    set(LIBCDS_WITH_HPX ON CACHE INTERNAL "")
    set(LIBCDS_INSIDE_HPX ON CACHE INTERNAL "")

    FetchContent_Declare(libcds
#            GIT_REPOSITORY https://github.com/khizmax/libcds
            GIT_REPOSITORY https://github.com/weilewei/libcds
            GIT_TAG hpx-thread
            GIT_SHALLOW TRUE
            )
    FetchContent_GetProperties(libcds)

    if(NOT libcds_POPULATED)
        FetchContent_Populate(libcds)
        add_subdirectory(${libcds_SOURCE_DIR} ${libcds_BINARY_DIR})
    endif()

endif()
