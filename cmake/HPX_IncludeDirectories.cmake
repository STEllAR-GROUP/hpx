# Copyright (c) 2011-2012 Bryce Adelstein-Lelbach
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

# FIXME: Put into a common script with HPX_LinkDirectories.cmake

set(HPX_INCLUDEDIRECTORIES_LOADED TRUE)

if(NOT HPX_UTILS_LOADED)
  include(HPX_Utils)
endif()

macro(hpx_include_directories)
  if(HPX_NO_INSTALL)
    # Don't use hpx_include_directories to avoid passing paths from the source
    # tree to FindHPX.cmake
    set(HPX_INCLUDE_DIRECTORIES ${HPX_INCLUDE_DIRECTORIES} ${ARGN})
  endif()
  include_directories(${ARGN})
endmacro()

macro(hpx_include_sys_directories)
  set(HPX_INCLUDE_DIRECTORIES ${HPX_INCLUDE_DIRECTORIES} ${ARGN})
  include_directories(${ARGN})
endmacro()

