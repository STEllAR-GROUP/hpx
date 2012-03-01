# Copyright (c) 2012 Bryce Adelstein-Lelbach
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

hpx_write_boostbook_catalog(${CMAKE_CURRENT_BINARY_DIR}/boostbook_catalog.xml)

file(GLOB HPX_QBK_FILES "*.qbk")
set(DOCUMENTATION_FILES
    ${HPX_QBK_FILES})

hpx_quickbook_to_html(hpx
    SOURCE hpx.qbk
    DEPENDENCIES ${DOCUMENTATION_FILES}
    CATALOG ${CMAKE_CURRENT_BINARY_DIR}/boostbook_catalog.xml
    XSLTPROC_ARGS --stringparam chunk.section.depth 4
                  --stringparam chunk.first.sections 1
                  --stringparam toc.section.depth 3
                  --stringparam toc.max.depth 3
                  --stringparam generate.section.toc.level 4
                  --stringparam boost.image.src images/hpx_0_8.png
                  --stringparam boost.image.w 323
                  --stringparam boost.image.h 68
                  --stringparam boost.image.alt "HPX - High Performance ParalleX")

add_custom_target(docs
    DEPENDS hpx_HTML.manifest
    SOURCES ${DOCUMENTATION_FILES})

source_group(Documentation
    FILES ${DOCUMENTATION_FILES})

set_source_files_properties(${DOCUMENTATION_FILES}
    PROPERTIES HEADER_FILE_ONLY TRUE)
