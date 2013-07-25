# Copyright (c) 2011 Bryce Lelbach
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

if(NOT HPX_FINDPATH_LOADED)
  include(HPX_FindPath)
endif()

if(DOCBOOK_USE_SYSTEM)
  set(DOCBOOK_DTD_USE_SYSTEM ON)
  set(DOCBOOK_XSL_USE_SYSTEM ON)
endif()

if(DOCBOOK_ROOT)
  set(DOCBOOK_DTD_ROOT ${DOCBOOK_ROOT})
  set(DOCBOOK_XSL_ROOT ${DOCBOOK_ROOT})
elseif(DEFINED ENV{DOCBOOK_ROOT})
  set (DOCBOOK_ROOT $ENV{DOCBOOK_ROOT})
  STRING(REPLACE \\ / DOCBOOK_ROOT ${DOCBOOK_ROOT})
  set(DOCBOOK_DTD_ROOT ${DOCBOOK_ROOT})
  set(DOCBOOK_XSL_ROOT ${DOCBOOK_ROOT})
endif()



hpx_find_path(DOCBOOK_DTD
  FILES docbookx.dtd
  FILE_PATHS
      share/xml/docbook/schema/dtd/4.2
      docbook-dtd share/sgml/docbook/xml-dtd-4.2)

hpx_find_path(DOCBOOK_XSL
  FILES html/html.xsl       # Do not move the html/ part into FILE_PATHS
        xhtml-1_1/html.xsl
  FILE_PATHS
      share/xml/docbook/stylesheet/docbook-xsl
      docbook-xsl
      share/sgml/docbook/xsl-stylesheets)

