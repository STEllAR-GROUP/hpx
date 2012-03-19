# Copyright (c) 2011-2012 Bryce Adelstein-Lelbach
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

# FIXME: Put into a common script with HPX_IncludeDirectories.cmake

set(HPX_LINKDIRECTORIES_LOADED TRUE)

if(NOT HPX_UTILS_LOADED)
  include(HPX_Utils)
endif()

macro(hpx_link_directories)
  set(HPX_LINK_DIRECTORIES ${HPX_LINK_DIRECTORIES} ${ARGN})
  link_directories(${ARGN})
endmacro()

