# Copyright (c) 2012 Bryce Adelstein-Lelbach
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

hpx_write_boostbook_catalog(${CMAKE_CURRENT_BINARY_DIR}/boostbook_catalog.xml)

file(GLOB base_files ${hpx_SOURCE_DIR}/docs/*.qbk)
file(GLOB_RECURSE manual_files ${hpx_SOURCE_DIR}/docs/manual/*.qbk)
file(GLOB_RECURSE tutorial_files ${hpx_SOURCE_DIR}/docs/tutorial/*.qbk)
file(GLOB_RECURSE index_files ${hpx_SOURCE_DIR}/docs/*.idx)

set(documentation_files
    ${manual_files} ${tutorial_files} ${base_files} ${index_files})

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
    ${hpx_SOURCE_DIR}/examples/hello_world_component/hello_world_component.cpp
    ${hpx_SOURCE_DIR}/examples/performance_counters/simplest_performance_counter.cpp
    ${hpx_SOURCE_DIR}/examples/performance_counters/sine/server/sine.hpp
    ${hpx_SOURCE_DIR}/hpx/performance_counters/performance_counter.hpp
    ${hpx_SOURCE_DIR}/hpx/performance_counters/base_performance_counter.hpp)

# Add all files here which should be passed to doxygen
set(doxygen_dependencies
    ${hpx_SOURCE_DIR}/hpx/hpx_fwd.hpp
    ${hpx_SOURCE_DIR}/hpx/hpx_init.hpp
    ${hpx_SOURCE_DIR}/hpx/hpx_start.hpp
    ${hpx_SOURCE_DIR}/hpx/hpx_finalize.hpp
    ${hpx_SOURCE_DIR}/hpx/exception.hpp
    ${hpx_SOURCE_DIR}/hpx/runtime/actions/action_support.hpp
    ${hpx_SOURCE_DIR}/hpx/runtime/actions/plain_action.hpp
    ${hpx_SOURCE_DIR}/hpx/runtime/actions/component_non_const_action.hpp
    ${hpx_SOURCE_DIR}/hpx/performance_counters/manage_counter_type.hpp)

foreach(doxygen_input ${doxygen_dependencies})
  set(doxygen_inputs "${doxygen_inputs} ${doxygen_input}")
endforeach()
#hpx_info("doxygen" "${doxygen_inputs}")

set(doxygen_output_file "${CMAKE_CURRENT_BINARY_DIR}/hpx_autodoc")
set(doxygen_output_dir "${CMAKE_CURRENT_BINARY_DIR}/doxygen")

set(doxygen_definition_list
    "DOXYGEN:=1"
    "BOOST_SCOPED_ENUM_START(x):=enum x"
    "BOOST_SCOPED_ENUM_END="
    "HPX_EXCEPTION_EXPORT="
    "HPX_EXPORT="
    "HPX_API_EXPORT="
    "HPX_STD_FUNCTION=std::function"
    "extern=")

#############################################################################
# Definitions for autodoc.doxy.in
set(doxygen_output_file "${CMAKE_CURRENT_BINARY_DIR}/hpx_autodoc")
set(doxygen_output_dir "${CMAKE_CURRENT_BINARY_DIR}/doxygen")

foreach(doxygen_predef ${doxygen_definition_list})
  set(doxygen_definitions "${doxygen_definitions} \"${doxygen_predef}\"")
endforeach()

foreach(doxygen_input ${doxygen_dependencies})
  set(doxygen_inputs "${doxygen_inputs} ${doxygen_input}")
endforeach()

hpx_info("doxygen" "Creating ${CMAKE_CURRENT_BINARY_DIR}/hpx_autodoc.doxy")
configure_file(${hpx_SOURCE_DIR}/cmake/templates/autodoc.doxy.in
                ${CMAKE_CURRENT_BINARY_DIR}/hpx_autodoc.doxy
                @ONLY)

#############################################################################
set(documentation_dependencies ${documentation_dependencies}
    ${CMAKE_CURRENT_BINARY_DIR}/hpx_autodoc.xml)

# Generate Doxygen from the source code.
hpx_source_to_boostbook(hpx_autodoc
    CATALOG ${CMAKE_CURRENT_BINARY_DIR}/boostbook_catalog.xml
    DEPENDENCIES ${doxygen_dependencies})

set(quickbook_fixme_mode "-D__hpx_disable_fixme__")
if(HPX_DOCUMENTATION_DEVELOPMENT)
  set(quickbook_fixme_mode "-D__hpx_enable_fixme__")
endif()

if(BOOSTAUTOINDEX_FOUND)
  set(auto_index_mode "-D__use_auto_index__")
  set(auto_index_xsl_param --stringparam index.on.type 1)
endif()

set(image
    images/hpx_${HPX_MAJOR_VERSION}_${HPX_MINOR_VERSION}_${HPX_PATCH_LEVEL}.png)

hpx_quickbook_to_html(hpx
  SOURCE hpx.qbk
  INDEX hpx.idx
  DEPENDENCIES ${documentation_dependencies}
  CATALOG ${CMAKE_CURRENT_BINARY_DIR}/boostbook_catalog.xml
  XSLTPROC_ARGS --stringparam chunk.section.depth 4
                --stringparam chunk.first.sections 1
                --stringparam toc.section.depth 4
                --stringparam toc.max.depth 4
                --stringparam generate.section.toc.level 4
                --stringparam boost.image.src ${image}
                --stringparam boost.image.w 458
                --stringparam boost.image.h 112
                --stringparam boost.image.alt "HPX - High Performance ParalleX"
                ${auto_index_xsl_param}
  QUICKBOOK_ARGS ${quickbook_fixme_mode} ${auto_index_mode}
  AUTOINDEX_ARGS --internal-index)

add_custom_target(docs
  DEPENDS hpx_HTML.manifest
  SOURCES ${documentation_files})

source_group(Documentation FILES ${base_files})
source_group("Documentation\\Manual" FILES ${manual_files})
source_group("Documentation\\Tutorial" FILES ${tutorial_files})
if(BOOSTAUTOINDEX_FOUND)
  source_group("Documentation\\Index" FILES ${index_files})
endif()

set_source_files_properties(${documentation_files}
  PROPERTIES HEADER_FILE_ONLY TRUE)

