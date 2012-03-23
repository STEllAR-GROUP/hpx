# Copyright (c) 2012 Bryce Adelstein-Lelbach
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

hpx_write_boostbook_catalog(${CMAKE_CURRENT_BINARY_DIR}/boostbook_catalog.xml)

file(GLOB HPX_QBK_FILES "*.qbk")
set(DOCUMENTATION_FILES
    ${HPX_QBK_FILES})

set(DOCUMENTATION_DEPENDENCIES
    ${DOCUMENTATION_FILES}
    ${hpx_SOURCE_DIR}/examples/quickstart/hello_world.cpp
    ${hpx_SOURCE_DIR}/examples/quickstart/simplest_hello_world.cpp
    ${hpx_SOURCE_DIR}/examples/accumulator/accumulators/server/managed_accumulator.hpp
    ${hpx_SOURCE_DIR}/examples/accumulator/accumulators/stubs/managed_accumulator.hpp
    ${hpx_SOURCE_DIR}/examples/accumulator/accumulators/managed_accumulator.hpp
    ${hpx_SOURCE_DIR}/examples/accumulator/accumulators/managed_accumulator.cpp
    ${hpx_SOURCE_DIR}/examples/accumulator/managed_accumulator_client.cpp
    ${hpx_SOURCE_DIR}/examples/hello_world_component/hello_world_component.hpp
    ${hpx_SOURCE_DIR}/examples/hello_world_component/hello_world_component.cpp)

set(quickbook_fixme_mode "-D__disable_fixme__")
if(HPX_DOCUMENTATION_DEVELOPMENT)
  set(quickbook_fixme_mode "-D__enable_fixme__")
endif()

hpx_quickbook_to_html(hpx
    SOURCE hpx.qbk
    DEPENDENCIES ${DOCUMENTATION_DEPENDENCIES}
    CATALOG ${CMAKE_CURRENT_BINARY_DIR}/boostbook_catalog.xml
    XSLTPROC_ARGS --stringparam chunk.section.depth 4
                  --stringparam chunk.first.sections 1
                  --stringparam toc.section.depth 3
                  --stringparam toc.max.depth 3
                  --stringparam generate.section.toc.level 4
                  --stringparam boost.image.src images/hpx_1_0.png
                  --stringparam boost.image.w 323
                  --stringparam boost.image.h 68
                  --stringparam boost.image.alt "HPX - High Performance ParalleX"
    QUICKBOOK_ARGS ${quickbook_fixme_mode})

add_custom_target(docs ALL
    DEPENDS hpx_HTML.manifest
    SOURCES ${DOCUMENTATION_FILES})

source_group(Documentation
    FILES ${DOCUMENTATION_FILES})

set_source_files_properties(${DOCUMENTATION_FILES}
    PROPERTIES HEADER_FILE_ONLY TRUE)
