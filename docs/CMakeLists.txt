# Copyright (c) 2012 Bryce Adelstein-Lelbach
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

hpx_write_boostbook_catalog(${CMAKE_CURRENT_BINARY_DIR}/boostbook_catalog.xml)

hpx_quickbook_to_html(guide
    SOURCE hpx.qbk
    CATALOG ${CMAKE_CURRENT_BINARY_DIR}/boostbook_catalog.xml
    XSLTPROC_ARGS --stringparam chunk.section.depth 4
                  --stringparam chunk.first.sections 1
                  --stringparam toc.section.depth 3
                  --stringparam toc.max.depth 3
                  --stringparam generate.section.toc.level 4)

add_custom_target(docs DEPENDS guide_HTML.manifest)

