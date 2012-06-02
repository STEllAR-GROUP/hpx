# Copyright (c) 2012 Bryce Adelstein-Lelbach
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

hpx_write_boostbook_catalog(${CMAKE_CURRENT_BINARY_DIR}/boostbook_catalog.xml)

file(GLOB base_files ${hpx_SOURCE_DIR}/docs/*.qbk)
file(GLOB_RECURSE manual_files ${hpx_SOURCE_DIR}/docs/manual/*.qbk)
file(GLOB_RECURSE tutorial_files ${hpx_SOURCE_DIR}/docs/tutorial/*.qbk)

set(documentation_files
    ${manual_files} ${tutorial_files} ${base_files})

set(documentation_dependencies
    ${documentation_files}
    ${hpx_SOURCE_DIR}/examples/quickstart/hello_world.cpp
    ${hpx_SOURCE_DIR}/examples/quickstart/simplest_hello_world.cpp
    ${hpx_SOURCE_DIR}/examples/quickstart/error_handling.cpp
    ${hpx_SOURCE_DIR}/examples/accumulator/accumulators/server/managed_accumulator.hpp
    ${hpx_SOURCE_DIR}/examples/accumulator/accumulators/stubs/managed_accumulator.hpp
    ${hpx_SOURCE_DIR}/examples/accumulator/accumulators/managed_accumulator.hpp
    ${hpx_SOURCE_DIR}/examples/accumulator/accumulators/managed_accumulator.cpp
    ${hpx_SOURCE_DIR}/examples/accumulator/managed_accumulator_client.cpp
    ${hpx_SOURCE_DIR}/examples/hello_world_component/hello_world_component.hpp
    ${hpx_SOURCE_DIR}/examples/hello_world_component/hello_world_component.cpp)

if(DOXYGEN_FOUND)
  set(documentation_dependencies ${documentation_dependencies}
      ${CMAKE_CURRENT_BINARY_DIR}/hpx_autodoc.xml)

  # Add all files here which should be passed to doxygen
  set(doxygen_dependencies
      ${hpx_SOURCE_DIR}/hpx/hpx_fwd.hpp
      ${hpx_SOURCE_DIR}/hpx/hpx_init.hpp
      ${hpx_SOURCE_DIR}/hpx/exception.hpp
      ${hpx_SOURCE_DIR}/hpx/runtime/actions/plain_action.hpp)

  # Generate Doxygen from the source code.
  hpx_source_to_boostbook(hpx_autodoc
    CATALOG ${CMAKE_CURRENT_BINARY_DIR}/boostbook_catalog.xml
    DEPENDENCIES ${doxygen_dependencies})
endif()

set(quickbook_fixme_mode "-D__hpx_disable_fixme__")
if(HPX_DOCUMENTATION_DEVELOPMENT)
  set(quickbook_fixme_mode "-D__hpx_enable_fixme__")
endif()

hpx_quickbook_to_html(hpx
  SOURCE hpx.qbk
  DEPENDENCIES ${documentation_dependencies}
  CATALOG ${CMAKE_CURRENT_BINARY_DIR}/boostbook_catalog.xml
  XSLTPROC_ARGS --stringparam chunk.section.depth 4
                --stringparam chunk.first.sections 1
                --stringparam toc.section.depth 4
                --stringparam toc.max.depth 4
                --stringparam generate.section.toc.level 4
                --stringparam boost.image.src images/hpx_0_9.png
                --stringparam boost.image.w 322
                --stringparam boost.image.h 85
                --stringparam boost.image.alt "HPX - High Performance ParalleX"
  QUICKBOOK_ARGS ${quickbook_fixme_mode} -D__hpx_svn_revision__=${SVN_REVISION})

add_custom_target(docs
  DEPENDS hpx_HTML.manifest
  SOURCES ${documentation_files})

source_group(Documentation FILES ${base_files})
source_group("Documentation\\Manual" FILES ${manual_files})
source_group("Documentation\\Tutorial" FILES ${tutorial_files})

set_source_files_properties(${documentation_files}
  PROPERTIES HEADER_FILE_ONLY TRUE)

