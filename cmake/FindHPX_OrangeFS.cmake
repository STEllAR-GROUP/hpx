# Copyright (c) 2013      Shuangyang Yang
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

if(NOT HPX_FINDPACKAGE_LOADED)
  include(HPX_FindPackage)
endif()

hpx_find_package(ORANGEFS
#  LIBRARIES  ofs orangefs pvfs2 orangefsposix
  LIBRARIES pvfs2 
  LIBRARY_PATHS lib 
  HEADERS pxfs.h orange.h 
  HEADER_PATHS include)

