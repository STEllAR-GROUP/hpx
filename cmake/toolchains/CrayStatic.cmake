# Copyright (c) 2014-2017 Thomas Heller
# Copyright (c) 2017      Bryce Adelstein Lelbach
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(HPX_WITH_STATIC_LINKING
    ON
    CACHE BOOL ""
)
set(HPX_WITH_STATIC_EXE_LINKING
    ON
    CACHE BOOL ""
)
set_property(GLOBAL PROPERTY TARGET_SUPPORTS_SHARED_LIBS FALSE)

# Set the Cray Compiler Wrapper
set(CMAKE_CXX_COMPILER CC)

set(CMAKE_CXX_FLAGS_INIT
    ""
    CACHE STRING ""
)
set(CMAKE_CXX_COMPILE_OBJECT
    "<CMAKE_CXX_COMPILER> -static -fPIC <DEFINES> <INCLUDES> <FLAGS> -o <OBJECT> -c <SOURCE>"
    CACHE STRING ""
)
set(CMAKE_CXX_LINK_EXECUTABLE
    "<CMAKE_CXX_COMPILER> -fPIC <FLAGS> <CMAKE_CXX_LINK_FLAGS> <LINK_FLAGS> <OBJECTS> -o <TARGET> <LINK_LIBRARIES>"
    CACHE STRING ""
)

# Disable searches in the default system paths. We are cross compiling after all
# and cmake might pick up wrong libraries that way
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM BOTH)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)

# We do a cross compilation here ...
set(CMAKE_CROSSCOMPILING
    ON
    CACHE BOOL ""
)

# RDTSCP is available on Xeon/Phis
set(HPX_WITH_RDTSCP
    ON
    CACHE BOOL ""
)

set(HPX_WITH_PARCELPORT_TCP
    ON
    CACHE BOOL ""
)
set(HPX_WITH_PARCELPORT_MPI
    ON
    CACHE BOOL ""
)
set(HPX_WITH_PARCELPORT_MPI_MULTITHREADED
    ON
    CACHE BOOL ""
)
