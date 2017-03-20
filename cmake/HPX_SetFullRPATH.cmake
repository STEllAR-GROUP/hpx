# Copyright (c) 2011-2014 Thomas Heller
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

if (NOT DEFINED LIB)
  set(LIB "lib")
endif(NOT DEFINED LIB)

list(FIND CMAKE_PLATFORM_IMPLICIT_LINK_DIRECTORIES "${CMAKE_INSTALL_PREFIX}/${LIB}" isSystemDir)
if(isSystemDir EQUAL -1)
  set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/${LIB}")
endif()
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH ON)
