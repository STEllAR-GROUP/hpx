# Copyright (c) 2012 Bryce Adelstein-Lelbach
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

hpx_write_boostbook_catalog("${CMAKE_CURRENT_BINARY_DIR}/boostbook_catalog.xml")

file(GLOB base_files "${PROJECT_SOURCE_DIR}/docs/*.qbk")
file(GLOB manual_files "${PROJECT_SOURCE_DIR}/docs/manual/*.qbk")
file(GLOB build_system_files "${PROJECT_SOURCE_DIR}/docs/manual/build_system/*.qbk")
file(GLOB_RECURSE tutorial_files "${PROJECT_SOURCE_DIR}/docs/tutorial/*.qbk")
file(GLOB_RECURSE index_files "${PROJECT_SOURCE_DIR}/docs/*.idx")

set(documentation_files
    ${manual_files} ${build_system_files} ${tutorial_files}
    ${base_files} ${index_files})

set(documentation_dependencies
    "${PROJECT_SOURCE_DIR}/examples/1d_stencil/1d_stencil_1.cpp"
    "${PROJECT_SOURCE_DIR}/examples/1d_stencil/1d_stencil_2.cpp"
    "${PROJECT_SOURCE_DIR}/examples/1d_stencil/1d_stencil_3.cpp"
    "${PROJECT_SOURCE_DIR}/examples/1d_stencil/1d_stencil_4.cpp"
    "${PROJECT_SOURCE_DIR}/examples/1d_stencil/1d_stencil_5.cpp"
    "${PROJECT_SOURCE_DIR}/examples/1d_stencil/1d_stencil_6.cpp"
    "${PROJECT_SOURCE_DIR}/examples/1d_stencil/1d_stencil_7.cpp"
    "${PROJECT_SOURCE_DIR}/examples/1d_stencil/1d_stencil_8.cpp"
    "${PROJECT_SOURCE_DIR}/examples/quickstart/hello_world.cpp"
    "${PROJECT_SOURCE_DIR}/examples/quickstart/simplest_hello_world.cpp"
    "${PROJECT_SOURCE_DIR}/examples/quickstart/error_handling.cpp"
    "${PROJECT_SOURCE_DIR}/examples/accumulators/server/accumulator.hpp"
    "${PROJECT_SOURCE_DIR}/examples/accumulators/accumulator.hpp"
    "${PROJECT_SOURCE_DIR}/examples/accumulators/accumulator.cpp"
    "${PROJECT_SOURCE_DIR}/examples/accumulators/accumulator_client.cpp"
    "${PROJECT_SOURCE_DIR}/examples/hello_world_component/hello_world_component.hpp"
    "${PROJECT_SOURCE_DIR}/examples/hello_world_component/hello_world_component.cpp"
    "${PROJECT_SOURCE_DIR}/examples/performance_counters/simplest_performance_counter.cpp"
    "${PROJECT_SOURCE_DIR}/examples/performance_counters/sine/server/sine.hpp"
    "${PROJECT_SOURCE_DIR}/hpx/performance_counters/performance_counter.hpp"
    "${PROJECT_SOURCE_DIR}/hpx/performance_counters/base_performance_counter.hpp")

set(all_documentation_dependencies
    ${documentation_files}
    ${documentation_dependencies})

# Add all files here which should be passed to doxygen
set(doxygen_dependencies
    "${PROJECT_SOURCE_DIR}/hpx/hpx_init.hpp"
    "${PROJECT_SOURCE_DIR}/hpx/hpx_start.hpp"
    "${PROJECT_SOURCE_DIR}/hpx/hpx_finalize.hpp"
    "${PROJECT_SOURCE_DIR}/hpx/error.hpp"
    "${PROJECT_SOURCE_DIR}/hpx/error_code.hpp"
    "${PROJECT_SOURCE_DIR}/hpx/exception.hpp"
    "${PROJECT_SOURCE_DIR}/hpx/exception_fwd.hpp"
    "${PROJECT_SOURCE_DIR}/hpx/exception_list.hpp"
    "${PROJECT_SOURCE_DIR}/hpx/throw_exception.hpp"
    "${PROJECT_SOURCE_DIR}/hpx/components/component_storage/migrate_from_storage.hpp"
    "${PROJECT_SOURCE_DIR}/hpx/components/component_storage/migrate_to_storage.hpp"
    "${PROJECT_SOURCE_DIR}/hpx/parallel/execution_policy.hpp"
    "${PROJECT_SOURCE_DIR}/hpx/parallel/algorithm.hpp"
    "${PROJECT_SOURCE_DIR}/hpx/parallel/task_block.hpp"
    "${PROJECT_SOURCE_DIR}/hpx/parallel/algorithms/adjacent_difference.hpp"
    "${PROJECT_SOURCE_DIR}/hpx/parallel/algorithms/adjacent_find.hpp"
    "${PROJECT_SOURCE_DIR}/hpx/parallel/algorithms/all_any_none.hpp"
    "${PROJECT_SOURCE_DIR}/hpx/parallel/algorithms/copy.hpp"
    "${PROJECT_SOURCE_DIR}/hpx/parallel/algorithms/count.hpp"
    "${PROJECT_SOURCE_DIR}/hpx/parallel/algorithms/destroy.hpp"
    "${PROJECT_SOURCE_DIR}/hpx/parallel/algorithms/equal.hpp"
    "${PROJECT_SOURCE_DIR}/hpx/parallel/algorithms/exclusive_scan.hpp"
    "${PROJECT_SOURCE_DIR}/hpx/parallel/algorithms/fill.hpp"
    "${PROJECT_SOURCE_DIR}/hpx/parallel/algorithms/find.hpp"
    "${PROJECT_SOURCE_DIR}/hpx/parallel/algorithms/for_each.hpp"
    "${PROJECT_SOURCE_DIR}/hpx/parallel/algorithms/for_loop.hpp"
    "${PROJECT_SOURCE_DIR}/hpx/parallel/algorithms/for_loop_induction.hpp"
    "${PROJECT_SOURCE_DIR}/hpx/parallel/algorithms/for_loop_reduction.hpp"
    "${PROJECT_SOURCE_DIR}/hpx/parallel/algorithms/generate.hpp"
    "${PROJECT_SOURCE_DIR}/hpx/parallel/algorithms/if_heap.hpp"
    "${PROJECT_SOURCE_DIR}/hpx/parallel/algorithms/includes.hpp"
    "${PROJECT_SOURCE_DIR}/hpx/parallel/algorithms/inclusive_scan.hpp"
    "${PROJECT_SOURCE_DIR}/hpx/parallel/algorithms/is_partitioned.hpp"
    "${PROJECT_SOURCE_DIR}/hpx/parallel/algorithms/is_sorted.hpp"
    "${PROJECT_SOURCE_DIR}/hpx/parallel/algorithms/lexicographical_compare.hpp"
    "${PROJECT_SOURCE_DIR}/hpx/parallel/algorithms/minmax.hpp"
    "${PROJECT_SOURCE_DIR}/hpx/parallel/algorithms/mismatch.hpp"
    "${PROJECT_SOURCE_DIR}/hpx/parallel/algorithms/move.hpp"
    "${PROJECT_SOURCE_DIR}/hpx/parallel/algorithms/reduce.hpp"
    "${PROJECT_SOURCE_DIR}/hpx/parallel/algorithms/reduce_by_key.hpp"
    "${PROJECT_SOURCE_DIR}/hpx/parallel/algorithms/remove_copy.hpp"
    "${PROJECT_SOURCE_DIR}/hpx/parallel/algorithms/replace.hpp"
    "${PROJECT_SOURCE_DIR}/hpx/parallel/algorithms/reverse.hpp"
    "${PROJECT_SOURCE_DIR}/hpx/parallel/algorithms/rotate.hpp"
    "${PROJECT_SOURCE_DIR}/hpx/parallel/algorithms/search.hpp"
    "${PROJECT_SOURCE_DIR}/hpx/parallel/algorithms/set_difference.hpp"
    "${PROJECT_SOURCE_DIR}/hpx/parallel/algorithms/set_intersection.hpp"
    "${PROJECT_SOURCE_DIR}/hpx/parallel/algorithms/set_symmetric_difference.hpp"
    "${PROJECT_SOURCE_DIR}/hpx/parallel/algorithms/set_union.hpp"
    "${PROJECT_SOURCE_DIR}/hpx/parallel/algorithms/sort.hpp"
    "${PROJECT_SOURCE_DIR}/hpx/parallel/algorithms/sort_by_key.hpp"
    "${PROJECT_SOURCE_DIR}/hpx/parallel/algorithms/swap_ranges.hpp"
    "${PROJECT_SOURCE_DIR}/hpx/parallel/algorithms/transform.hpp"
    "${PROJECT_SOURCE_DIR}/hpx/parallel/algorithms/transform_exclusive_scan.hpp"
    "${PROJECT_SOURCE_DIR}/hpx/parallel/algorithms/transform_inclusive_scan.hpp"
    "${PROJECT_SOURCE_DIR}/hpx/parallel/algorithms/transform_reduce.hpp"
    "${PROJECT_SOURCE_DIR}/hpx/parallel/algorithms/transform_reduce_binary.hpp"
    "${PROJECT_SOURCE_DIR}/hpx/parallel/algorithms/uninitialized_copy.hpp"
    "${PROJECT_SOURCE_DIR}/hpx/parallel/algorithms/uninitialized_fill.hpp"
    "${PROJECT_SOURCE_DIR}/hpx/parallel/container_algorithms/copy.hpp"
    "${PROJECT_SOURCE_DIR}/hpx/parallel/container_algorithms/for_each.hpp"
    "${PROJECT_SOURCE_DIR}/hpx/parallel/container_algorithms/generate.hpp"
    "${PROJECT_SOURCE_DIR}/hpx/parallel/container_algorithms/minmax.hpp"
    "${PROJECT_SOURCE_DIR}/hpx/parallel/container_algorithms/remove_copy.hpp"
    "${PROJECT_SOURCE_DIR}/hpx/parallel/container_algorithms/replace.hpp"
    "${PROJECT_SOURCE_DIR}/hpx/parallel/container_algorithms/reverse.hpp"
    "${PROJECT_SOURCE_DIR}/hpx/parallel/container_algorithms/rotate.hpp"
    "${PROJECT_SOURCE_DIR}/hpx/parallel/container_algorithms/sort.hpp"
    "${PROJECT_SOURCE_DIR}/hpx/parallel/container_algorithms/transform.hpp"
    "${PROJECT_SOURCE_DIR}/hpx/parallel/executors/auto_chunk_size.hpp"
    "${PROJECT_SOURCE_DIR}/hpx/parallel/executors/dynamic_chunk_size.hpp"
    "${PROJECT_SOURCE_DIR}/hpx/parallel/executors/executor_traits.hpp"
    "${PROJECT_SOURCE_DIR}/hpx/parallel/executors/executor_parameter_traits.hpp"
    "${PROJECT_SOURCE_DIR}/hpx/parallel/executors/guided_chunk_size.hpp"
    "${PROJECT_SOURCE_DIR}/hpx/parallel/executors/parallel_executor.hpp"
    "${PROJECT_SOURCE_DIR}/hpx/parallel/executors/persistent_auto_chunk_size.hpp"
    "${PROJECT_SOURCE_DIR}/hpx/parallel/executors/sequenced_executor.hpp"
    "${PROJECT_SOURCE_DIR}/hpx/parallel/executors/service_executors.hpp"
    "${PROJECT_SOURCE_DIR}/hpx/parallel/executors/static_chunk_size.hpp"
    "${PROJECT_SOURCE_DIR}/hpx/parallel/executors/thread_pool_executors.hpp"
    "${PROJECT_SOURCE_DIR}/hpx/parallel/executors/timed_executor_traits.hpp"
    "${PROJECT_SOURCE_DIR}/hpx/runtime_fwd.hpp"
    "${PROJECT_SOURCE_DIR}/hpx/runtime/applier_fwd.hpp"
    "${PROJECT_SOURCE_DIR}/hpx/runtime/basename_registration.hpp"
    "${PROJECT_SOURCE_DIR}/hpx/runtime/find_here.hpp"
    "${PROJECT_SOURCE_DIR}/hpx/runtime/find_localities.hpp"
    "${PROJECT_SOURCE_DIR}/hpx/runtime/get_colocation_id.hpp"
    "${PROJECT_SOURCE_DIR}/hpx/runtime/get_ptr.hpp"
    "${PROJECT_SOURCE_DIR}/hpx/runtime/get_locality_id.hpp"
    "${PROJECT_SOURCE_DIR}/hpx/runtime/get_locality_name.hpp"
    "${PROJECT_SOURCE_DIR}/hpx/runtime/get_num_localities.hpp"
    "${PROJECT_SOURCE_DIR}/hpx/runtime/get_os_thread_count.hpp"
    "${PROJECT_SOURCE_DIR}/hpx/runtime/get_thread_name.hpp"
    "${PROJECT_SOURCE_DIR}/hpx/runtime/get_worker_thread_num.hpp"
    "${PROJECT_SOURCE_DIR}/hpx/runtime/launch_policy.hpp"
    "${PROJECT_SOURCE_DIR}/hpx/runtime/report_error.hpp"
    "${PROJECT_SOURCE_DIR}/hpx/runtime/runtime_mode.hpp"
    "${PROJECT_SOURCE_DIR}/hpx/runtime/set_parcel_write_handler.hpp"
    "${PROJECT_SOURCE_DIR}/hpx/runtime/shutdown_function.hpp"
    "${PROJECT_SOURCE_DIR}/hpx/runtime/startup_function.hpp"
    "${PROJECT_SOURCE_DIR}/hpx/runtime/trigger_lco.hpp"
    "${PROJECT_SOURCE_DIR}/hpx/runtime/actions/basic_action.hpp"
    "${PROJECT_SOURCE_DIR}/hpx/runtime/actions/component_action.hpp"
    "${PROJECT_SOURCE_DIR}/hpx/runtime/actions/plain_action.hpp"
    "${PROJECT_SOURCE_DIR}/hpx/runtime/components/binpacking_distribution_policy.hpp"
    "${PROJECT_SOURCE_DIR}/hpx/runtime/components/colocating_distribution_policy.hpp"
    "${PROJECT_SOURCE_DIR}/hpx/runtime/components/component_factory.hpp"
    "${PROJECT_SOURCE_DIR}/hpx/runtime/components/copy_component.hpp"
    "${PROJECT_SOURCE_DIR}/hpx/runtime/components/default_distribution_policy.hpp"
    "${PROJECT_SOURCE_DIR}/hpx/runtime/components/migrate_component.hpp"
    "${PROJECT_SOURCE_DIR}/hpx/runtime/components/new.hpp"
    "${PROJECT_SOURCE_DIR}/hpx/runtime/naming/unmanaged.hpp"
    "${PROJECT_SOURCE_DIR}/hpx/runtime/threads/thread_enums.hpp"
    "${PROJECT_SOURCE_DIR}/hpx/runtime/threads/thread_data_fwd.hpp"
    "${PROJECT_SOURCE_DIR}/hpx/runtime/threads/thread_helpers.hpp"
    "${PROJECT_SOURCE_DIR}/hpx/lcos/barrier.hpp"
    "${PROJECT_SOURCE_DIR}/hpx/lcos/broadcast.hpp"
    "${PROJECT_SOURCE_DIR}/hpx/lcos/fold.hpp"
    "${PROJECT_SOURCE_DIR}/hpx/lcos/gather.hpp"
    "${PROJECT_SOURCE_DIR}/hpx/lcos/reduce.hpp"
    "${PROJECT_SOURCE_DIR}/hpx/lcos/split_future.hpp"
    "${PROJECT_SOURCE_DIR}/hpx/lcos/wait_all.hpp"
    "${PROJECT_SOURCE_DIR}/hpx/lcos/when_all.hpp"
    "${PROJECT_SOURCE_DIR}/hpx/lcos/wait_any.hpp"
    "${PROJECT_SOURCE_DIR}/hpx/lcos/when_any.hpp"
    "${PROJECT_SOURCE_DIR}/hpx/lcos/wait_some.hpp"
    "${PROJECT_SOURCE_DIR}/hpx/lcos/when_some.hpp"
    "${PROJECT_SOURCE_DIR}/hpx/lcos/wait_each.hpp"
    "${PROJECT_SOURCE_DIR}/hpx/lcos/when_each.hpp"
    "${PROJECT_SOURCE_DIR}/hpx/util/invoke.hpp"
    "${PROJECT_SOURCE_DIR}/hpx/util/invoke_fused.hpp"
    "${PROJECT_SOURCE_DIR}/hpx/performance_counters/manage_counter_type.hpp")


foreach(doxygen_input ${doxygen_dependencies})
  set(doxygen_inputs "${doxygen_inputs} ${doxygen_input}")
endforeach()
#hpx_info("doxygen" "${doxygen_inputs}")

set(doxygen_output_file "${CMAKE_CURRENT_BINARY_DIR}/hpx_autodoc")
set(doxygen_output_dir "${CMAKE_CURRENT_BINARY_DIR}/doxygen")

set(doxygen_definition_list
    "DOXYGEN:=1"
    "BOOST_SYSTEM_NOEXCEPT="
    "HPX_EXCEPTION_EXPORT="
    "HPX_EXPORT="
    "HPX_API_EXPORT="
    "HPX_ALWAYS_EXPORT="
    "extern="
    "HPX_NOEXCEPT="
    "HPX_CONSTEXPR="
    "HPX_FORCEINLINE="
    "HPX_INLINE_NAMESPACE(x)=namespace x"
    "HPX_CONCEPT_REQUIRES_(x)=")

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

hpx_info("Creating Doxyfile: ${CMAKE_CURRENT_BINARY_DIR}/hpx_autodoc.doxy")
configure_file("${PROJECT_SOURCE_DIR}/cmake/templates/autodoc.doxy.in"
                "${CMAKE_CURRENT_BINARY_DIR}/hpx_autodoc.doxy"
                @ONLY)

#############################################################################
set(all_documentation_dependencies ${all_documentation_dependencies}
    ${CMAKE_CURRENT_BINARY_DIR}/hpx_autodoc.xml)

# Generate Doxygen from the source code.
hpx_source_to_boostbook(hpx_autodoc
    CATALOG "${CMAKE_CURRENT_BINARY_DIR}/boostbook_catalog.xml"
    DEPENDENCIES ${doxygen_dependencies})

set(quickbook_fixme_mode "-D__hpx_disable_fixme__")
if(HPX_DOCUMENTATION_DEVELOPMENT)
  set(quickbook_fixme_mode "-D__hpx_enable_fixme__")
endif()

if(BOOSTAUTOINDEX_FOUND)
  set(auto_index_mode "-D__use_auto_index__")
  set(auto_index_xsl_param index.on.type=1)
endif()

set(image
    images/hpx_${HPX_MAJOR_VERSION}_${HPX_MINOR_VERSION}_${HPX_PATCH_LEVEL}_draft.png)

set(HPX_CMAKE_OPTIONS "")
foreach(_cat ${HPX_OPTION_CATEGORIES})
  string(REPLACE " " "_" _cat_link "${_cat}")
  set(HPX_CMAKE_OPTIONS
    "${HPX_CMAKE_OPTIONS}* [link build_system.cmake_variables.${_cat_link} ${_cat} Options]\n")
endforeach()
set(HPX_CMAKE_OPTIONS "${HPX_CMAKE_OPTIONS}\n")

get_cmake_property(_variableNames CACHE_VARIABLES)
foreach(_cat ${HPX_OPTION_CATEGORIES})
  string(REPLACE " " "_" _cat_anchor "${_cat}")
  set(HPX_CMAKE_OPTIONS
    "${HPX_CMAKE_OPTIONS}[#build_system.cmake_variables.${_cat_anchor}][h3 ${_cat} Options]\n")
  foreach (_variableName ${_variableNames})
    if(${_variableName}Category)
      if(_cat STREQUAL ${_variableName}Category)
        string(REPLACE " " "_" _variableName_anchor "${_variableName}")
        set(HPX_CMAKE_OPTIONS
          "${HPX_CMAKE_OPTIONS}* [link build_system.cmake_variables.${_variableName_anchor} ${_variableName}]\n")
      endif()
    endif()
  endforeach()
  set(HPX_CMAKE_OPTIONS "${HPX_CMAKE_OPTIONS}\n")
  set(HPX_CMAKE_OPTIONS "${HPX_CMAKE_OPTIONS}[variablelist\n")

  foreach (_variableName ${_variableNames})
    if(${_variableName}Category)
      if(_cat STREQUAL ${_variableName}Category)
        string(REPLACE " " "_" _variableName_anchor "${_variableName}")

        get_property(_doc CACHE "${_variableName}" PROPERTY HELPSTRING)
        get_property(_type CACHE "${_variableName}" PROPERTY TYPE)
        if(NOT _type STREQUAL "UNINITIALIZED")
          set(_variableName "${_variableName}:${_type}")
        endif()
        set(HPX_CMAKE_OPTIONS
          "${HPX_CMAKE_OPTIONS}    ")
        set(HPX_CMAKE_OPTIONS
          "${HPX_CMAKE_OPTIONS}    [[[#build_system.cmake_variables.${_variableName_anchor}] `${_variableName}`][${_doc}]]\n")
      endif()
    endif()
  endforeach()

  set(HPX_CMAKE_OPTIONS "${HPX_CMAKE_OPTIONS}] [/ ${_cat} Options]\n\n")
endforeach()

set(QUICKBOOK_CMAKE_VARIABLES_DEST
    "${CMAKE_CURRENT_BINARY_DIR}/generated/build_system/cmake_variables.qbk")
configure_file(
  "${PROJECT_SOURCE_DIR}/cmake/templates/cmake_variables.qbk.in"
  ${QUICKBOOK_CMAKE_VARIABLES_DEST}
  @ONLY
)

set(toolchains_base_dir ${PROJECT_SOURCE_DIR}/cmake/toolchains/)
file(GLOB _toolchain_files ${toolchains_base_dir} ${toolchains_base_dir}*.cmake)

set(HPX_CMAKE_TOOLCHAINS_LIST "")
set(HPX_CMAKE_TOOLCHAINS "")
foreach(_toolchain ${_toolchain_files})
  get_filename_component(_toolchain_name ${_toolchain} NAME_WE)
  string(REGEX REPLACE "[ -]" "_" _toolchain_anchor "${_toolchain_name}")
  set(HPX_CMAKE_TOOLCHAINS_LIST
   "${HPX_CMAKE_TOOLCHAINS_LIST}* [link hpx.manual.build_system.building_hpx.cmake_toolchains.${_toolchain_anchor} ${_toolchain_name}]\n")
  set(HPX_CMAKE_TOOLCHAINS
    "${HPX_CMAKE_TOOLCHAINS}[heading:${_toolchain_anchor} ${_toolchain_name}]\n")
  set(HPX_CMAKE_TOOLCHAINS "${HPX_CMAKE_TOOLCHAINS}\n``\n")
  file(STRINGS ${_toolchain} _toolchain_content)
  foreach(_line ${_toolchain_content})
    set(HPX_CMAKE_TOOLCHAINS "${HPX_CMAKE_TOOLCHAINS}    ${_line}\n")
  endforeach()
  set(HPX_CMAKE_TOOLCHAINS "${HPX_CMAKE_TOOLCHAINS}``\n")
  set(HPX_CMAKE_TOOLCHAINS "${HPX_CMAKE_TOOLCHAINS}\n")
endforeach()

set(QUICKBOOK_CMAKE_TOOLCHAINS_DEST
    "${CMAKE_CURRENT_BINARY_DIR}/generated/build_system/cmake_toolchains.qbk")
configure_file(
  "${PROJECT_SOURCE_DIR}/cmake/templates/cmake_toolchains.qbk.in"
  ${QUICKBOOK_CMAKE_TOOLCHAINS_DEST}
  @ONLY
)

hpx_quickbook_to_html(hpx
  SOURCE hpx.qbk
  INDEX hpx.idx
  DEPENDENCIES
    ${all_documentation_dependencies}
    ${QUICKBOOK_CMAKE_VARIABLES_DEST}
    ${QUICKBOOK_CMAKE_TOOLCHAINS_DEST}
  CATALOG "${CMAKE_CURRENT_BINARY_DIR}/boostbook_catalog.xml"
  XSLTPROC_ARGS chunk.section.depth=4
                chunk.first.sections=1
                toc.section.depth=4
                toc.max.depth=4
                generate.section.toc.level=4
                boost.image.src=${image}
                boost.image.w=531
                boost.image.h=105
                boost.image.alt="HPX"
                ${auto_index_xsl_param}
  QUICKBOOK_ARGS ${quickbook_fixme_mode} ${auto_index_mode}
  AUTOINDEX_ARGS --internal-index
  SINGLEPAGE ${HPX_WITH_DOCUMENTATION_SINGLEPAGE})

if(FOP_FOUND)
  # This assumes that Quickbook -> BoostBook succeeded above.
#  hpx_quickbook_to_boostbook(hpx_pdf SOURCE hpx.qbk
#    DEPENDENCIES ${all_documentation_dependencies}
#    QUICKBOOK_ARGS ${quickbook_fixme_mode} ${auto_index_mode}
#    NODOXYGEN)

  if(BOOSTAUTOINDEX_FOUND)
    hpx_docbook_to_xslfo(hpx SOURCE hpx_auto_index.dbk
                         XSLTPROC_ARGS index.on.type=1)
  else()
    hpx_docbook_to_xslfo(hpx SOURCE hpx.dbk)
  endif()

  hpx_xslfo_to_pdf(hpx SOURCE hpx.fo)
endif()

if(FOP_FOUND)
  add_custom_target(docs
    DEPENDS hpx_HTML.manifest hpx.pdf
    SOURCES ${documentation_files})
else()
  add_custom_target(docs
    DEPENDS hpx_HTML.manifest
    SOURCES ${documentation_files})
endif()

set_target_properties(docs
    PROPERTIES FOLDER "Documentation/")

if(HPX_WITH_DOCUMENTATION_SINGLEPAGE)
  add_custom_target(singlepage_docs
    DEPENDS hpx_singlepage_HTML.manifest
    SOURCES ${documentation_files})
  set_target_properties(singlepage_docs
      PROPERTIES FOLDER "Documentation/")
endif()

# Create links to properly view the docs in the build directory
set(DOCS_OUTPUT_DIR "${CMAKE_BINARY_DIR}/share/hpx-${HPX_VERSION}/docs/html")

execute_process(COMMAND "${CMAKE_COMMAND}" -E make_directory "${DOCS_OUTPUT_DIR}" )
execute_process(COMMAND "${CMAKE_COMMAND}" -E make_directory "${DOCS_OUTPUT_DIR}/code" )

create_symbolic_link("${PROJECT_SOURCE_DIR}/docs/html/src"
                     "${DOCS_OUTPUT_DIR}/src")
create_symbolic_link("${PROJECT_SOURCE_DIR}/docs/html/images"
                     "${DOCS_OUTPUT_DIR}/images")
create_symbolic_link("${PROJECT_SOURCE_DIR}/hpx"
                     "${DOCS_OUTPUT_DIR}/code/hpx")
create_symbolic_link("${PROJECT_SOURCE_DIR}/src"
                     "${DOCS_OUTPUT_DIR}/code/src")
create_symbolic_link("${PROJECT_SOURCE_DIR}/examples"
                     "${DOCS_OUTPUT_DIR}/code/examples")
create_symbolic_link("${PROJECT_SOURCE_DIR}/tests"
                     "${DOCS_OUTPUT_DIR}/code/tests")

source_group(Documentation FILES ${base_files})
source_group("Documentation\\Manual" FILES ${manual_files})
source_group("Documentation\\Manual\\BuildSystem" FILES ${build_system_files})
source_group("Documentation\\Tutorial" FILES ${tutorial_files})
if(BOOSTAUTOINDEX_FOUND)
  source_group("Documentation\\Index" FILES ${index_files})
endif()

set_source_files_properties(${documentation_files}
  PROPERTIES HEADER_FILE_ONLY TRUE)

###############################################################################
# handle generating docs for github pages
add_custom_target(git_docs
  COMMAND "${CMAKE_COMMAND}"
    -DHPX_SOURCE_DIR:PATH=${CMAKE_SOURCE_DIR}
    -DHPX_VERSION:STRING=${HPX_VERSION}
    -DHPX_DOCUMENTATION_FILES="${documentation_dependencies};${doxygen_dependencies}"
    -P "${CMAKE_SOURCE_DIR}/cmake/HPX_UpdateGitDocs.cmake")
#  VERBATIM)

set_target_properties(git_docs
    PROPERTIES FOLDER "Documentation/")

add_dependencies(git_docs docs)
if(HPX_WITH_DOCUMENTATION_SINGLEPAGE)
  add_dependencies(git_docs singlepage_docs)
endif()

