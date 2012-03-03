# Copyright (c) 2011 Bryce Lelbach
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

if(NOT HPX_FINDPATH_LOADED)
  include(HPX_FindPath)
endif()

if(BOOST_USE_SYSTEM)
  set(BOOSTBOOK_DTD_USE_SYSTEM ON)
  set(BOOSTBOOK_XSL_USE_SYSTEM ON)
endif()

if(NOT BOOSTBOOK_ROOT)
  if(BOOST_ROOT)
    set(BOOSTBOOK_ROOT ${BOOST_ROOT})
  elseif($ENV{BOOST_ROOT})
    set(BOOSTBOOK_ROOT $ENV{BOOST_ROOT})
  endif()
endif()

if(BOOSTBOOK_ROOT)
  set(BOOSTBOOK_DTD_ROOT ${BOOSTBOOK_ROOT})
  set(BOOSTBOOK_XSL_ROOT ${BOOSTBOOK_ROOT})
endif()

hpx_find_path(BOOSTBOOK_DTD
  FILES boostbook.dtd
  FILE_PATHS share/boostbook/dtd dist/share/boostbook/dtd tools/boostbook/dtd)

hpx_find_path(BOOSTBOOK_XSL
  FILES docbook.xsl
  FILE_PATHS share/boostbook/xsl dist/share/boostbook/xsl tools/boostbook/xsl)

