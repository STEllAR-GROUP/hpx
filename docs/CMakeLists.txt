# Copyright (c) 2012 Bryce Adelstein-Lelbach
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

# Add all files here which should be passed to doxygen
set(doxygen_dependencies
    "${PROJECT_SOURCE_DIR}/hpx/hpx_init.hpp"
    "${PROJECT_SOURCE_DIR}/hpx/hpx_start.hpp"
    "${PROJECT_SOURCE_DIR}/hpx/hpx_finalize.hpp"
    "${PROJECT_SOURCE_DIR}/hpx/hpx_suspend.hpp"
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
    "${PROJECT_SOURCE_DIR}/hpx/parallel/algorithms/is_heap.hpp"
    "${PROJECT_SOURCE_DIR}/hpx/parallel/algorithms/includes.hpp"
    "${PROJECT_SOURCE_DIR}/hpx/parallel/algorithms/inclusive_scan.hpp"
    "${PROJECT_SOURCE_DIR}/hpx/parallel/algorithms/is_partitioned.hpp"
    "${PROJECT_SOURCE_DIR}/hpx/parallel/algorithms/is_sorted.hpp"
    "${PROJECT_SOURCE_DIR}/hpx/parallel/algorithms/lexicographical_compare.hpp"
    "${PROJECT_SOURCE_DIR}/hpx/parallel/algorithms/merge.hpp"
    "${PROJECT_SOURCE_DIR}/hpx/parallel/algorithms/minmax.hpp"
    "${PROJECT_SOURCE_DIR}/hpx/parallel/algorithms/mismatch.hpp"
    "${PROJECT_SOURCE_DIR}/hpx/parallel/algorithms/move.hpp"
    "${PROJECT_SOURCE_DIR}/hpx/parallel/algorithms/partition.hpp"
    "${PROJECT_SOURCE_DIR}/hpx/parallel/algorithms/reduce.hpp"
    "${PROJECT_SOURCE_DIR}/hpx/parallel/algorithms/reduce_by_key.hpp"
    "${PROJECT_SOURCE_DIR}/hpx/parallel/algorithms/remove.hpp"
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
    "${PROJECT_SOURCE_DIR}/hpx/parallel/algorithms/uninitialized_default_construct.hpp"
    "${PROJECT_SOURCE_DIR}/hpx/parallel/algorithms/uninitialized_fill.hpp"
    "${PROJECT_SOURCE_DIR}/hpx/parallel/algorithms/uninitialized_move.hpp"
    "${PROJECT_SOURCE_DIR}/hpx/parallel/algorithms/uninitialized_value_construct.hpp"
    "${PROJECT_SOURCE_DIR}/hpx/parallel/algorithms/unique.hpp"
    "${PROJECT_SOURCE_DIR}/hpx/parallel/container_algorithms/all_any_none.hpp"
    "${PROJECT_SOURCE_DIR}/hpx/parallel/container_algorithms/copy.hpp"
    "${PROJECT_SOURCE_DIR}/hpx/parallel/container_algorithms/count.hpp"
    "${PROJECT_SOURCE_DIR}/hpx/parallel/container_algorithms/fill.hpp"
    "${PROJECT_SOURCE_DIR}/hpx/parallel/container_algorithms/find.hpp"
    "${PROJECT_SOURCE_DIR}/hpx/parallel/container_algorithms/for_each.hpp"
    "${PROJECT_SOURCE_DIR}/hpx/parallel/container_algorithms/generate.hpp"
    "${PROJECT_SOURCE_DIR}/hpx/parallel/container_algorithms/is_heap.hpp"
    "${PROJECT_SOURCE_DIR}/hpx/parallel/container_algorithms/merge.hpp"
    "${PROJECT_SOURCE_DIR}/hpx/parallel/container_algorithms/minmax.hpp"
    "${PROJECT_SOURCE_DIR}/hpx/parallel/container_algorithms/partition.hpp"
    "${PROJECT_SOURCE_DIR}/hpx/parallel/container_algorithms/remove.hpp"
    "${PROJECT_SOURCE_DIR}/hpx/parallel/container_algorithms/remove_copy.hpp"
    "${PROJECT_SOURCE_DIR}/hpx/parallel/container_algorithms/replace.hpp"
    "${PROJECT_SOURCE_DIR}/hpx/parallel/container_algorithms/reverse.hpp"
    "${PROJECT_SOURCE_DIR}/hpx/parallel/container_algorithms/rotate.hpp"
    "${PROJECT_SOURCE_DIR}/hpx/parallel/container_algorithms/search.hpp"
    "${PROJECT_SOURCE_DIR}/hpx/parallel/container_algorithms/sort.hpp"
    "${PROJECT_SOURCE_DIR}/hpx/parallel/container_algorithms/transform.hpp"
    "${PROJECT_SOURCE_DIR}/hpx/parallel/container_algorithms/unique.hpp"
    "${PROJECT_SOURCE_DIR}/hpx/parallel/executors/auto_chunk_size.hpp"
    "${PROJECT_SOURCE_DIR}/hpx/parallel/executors/dynamic_chunk_size.hpp"
    "${PROJECT_SOURCE_DIR}/hpx/parallel/executors/execution_fwd.hpp"
    "${PROJECT_SOURCE_DIR}/hpx/parallel/executors/execution_information_fwd.hpp"
    "${PROJECT_SOURCE_DIR}/hpx/parallel/executors/guided_chunk_size.hpp"
    "${PROJECT_SOURCE_DIR}/hpx/parallel/executors/parallel_executor.hpp"
    "${PROJECT_SOURCE_DIR}/hpx/parallel/executors/persistent_auto_chunk_size.hpp"
    "${PROJECT_SOURCE_DIR}/hpx/parallel/executors/sequenced_executor.hpp"
    "${PROJECT_SOURCE_DIR}/hpx/parallel/executors/service_executors.hpp"
    "${PROJECT_SOURCE_DIR}/hpx/parallel/executors/static_chunk_size.hpp"
    "${PROJECT_SOURCE_DIR}/hpx/parallel/executors/thread_pool_executors.hpp"
    "${PROJECT_SOURCE_DIR}/hpx/performance_counters/manage_counter_type.hpp"
    "${PROJECT_SOURCE_DIR}/hpx/runtime_fwd.hpp"
    "${PROJECT_SOURCE_DIR}/hpx/runtime/applier_fwd.hpp"
    "${PROJECT_SOURCE_DIR}/hpx/runtime/basename_registration_fwd.hpp"
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
    "${PROJECT_SOURCE_DIR}/hpx/runtime/resource/partitioner.hpp"
    "${PROJECT_SOURCE_DIR}/hpx/runtime/resource/partitioner_fwd.hpp"
    "${PROJECT_SOURCE_DIR}/hpx/runtime/threads/policies/scheduler_mode.hpp"
    "${PROJECT_SOURCE_DIR}/hpx/runtime/threads/thread_enums.hpp"
    "${PROJECT_SOURCE_DIR}/hpx/runtime/threads/thread_data_fwd.hpp"
    "${PROJECT_SOURCE_DIR}/hpx/runtime/threads/thread_helpers.hpp"
    "${PROJECT_SOURCE_DIR}/hpx/runtime/threads/thread_pool_base.hpp"
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
    "${PROJECT_SOURCE_DIR}/hpx/traits/is_execution_policy.hpp"
    "${PROJECT_SOURCE_DIR}/hpx/util/checkpoint.hpp"
    "${PROJECT_SOURCE_DIR}/hpx/util/debugging.hpp"
    "${PROJECT_SOURCE_DIR}/hpx/util/invoke.hpp"
    "${PROJECT_SOURCE_DIR}/hpx/util/invoke_fused.hpp"
    "${PROJECT_SOURCE_DIR}/hpx/util/pack_traversal.hpp"
    "${PROJECT_SOURCE_DIR}/hpx/util/pack_traversal_async.hpp"
    "${PROJECT_SOURCE_DIR}/hpx/util/unwrap.hpp"
    "${PROJECT_SOURCE_DIR}/hpx/util/unwrapped.hpp"
    "${PROJECT_SOURCE_DIR}/hpx/performance_counters/manage_counter_type.hpp")

foreach(doxygen_input ${doxygen_dependencies})
  set(doxygen_inputs "${doxygen_inputs} ${doxygen_input}")
endforeach()

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
    "HPX_CONCEPT_REQUIRES_(...)=")

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

# Generate rst files with CMake variables and toolchains
set(HPX_CMAKE_OPTIONS_RST "")
foreach(_cat ${HPX_OPTION_CATEGORIES})
  string(REPLACE " " "_" _cat_link "${_cat}")
  set(HPX_CMAKE_OPTIONS_RST
    "${HPX_CMAKE_OPTIONS_RST}* :ref:`${_cat} options <${_cat_link}>`\n")
endforeach()
set(HPX_CMAKE_OPTIONS_RST "${HPX_CMAKE_OPTIONS_RST}\n")

get_cmake_property(_variableNames CACHE_VARIABLES)
foreach(_cat ${HPX_OPTION_CATEGORIES})
  string(REPLACE " " "_" _cat_anchor "${_cat}")
  set(_cat_title "${_cat} options")
  string(REGEX REPLACE "." "-" _cat_title_underline "${_cat_title}")
  set(HPX_CMAKE_OPTIONS_RST
    "${HPX_CMAKE_OPTIONS_RST}.. _${_cat_anchor}:\n\n${_cat_title}\n${_cat_title_underline}\n\n")
  foreach (_variableName ${_variableNames})
    if(${_variableName}Category)
      if(_cat STREQUAL ${_variableName}Category)
        string(REPLACE " " "_" _variableName_anchor "${_variableName}")

        get_property(_doc CACHE "${_variableName}" PROPERTY HELPSTRING)
        get_property(_type CACHE "${_variableName}" PROPERTY TYPE)
        if(NOT _type STREQUAL "UNINITIALIZED")
          set(_variableName "${_variableName}:${_type}")
        endif()
        set(HPX_CMAKE_OPTIONS_RST
          "${HPX_CMAKE_OPTIONS_RST}* :option:`${_variableName}`\n")
      endif()
    endif()
  endforeach()
  set(HPX_CMAKE_OPTIONS_RST "${HPX_CMAKE_OPTIONS_RST}\n")

  foreach (_variableName ${_variableNames})
    if(${_variableName}Category)
      if(_cat STREQUAL ${_variableName}Category)
        string(REPLACE " " "_" _variableName_anchor "${_variableName}")

        get_property(_doc CACHE "${_variableName}" PROPERTY HELPSTRING)
        get_property(_type CACHE "${_variableName}" PROPERTY TYPE)
        if(NOT _type STREQUAL "UNINITIALIZED")
          set(_variableName "${_variableName}:${_type}")
        endif()
        set(HPX_CMAKE_OPTIONS_RST
          "${HPX_CMAKE_OPTIONS_RST}.. option:: ${_variableName}\n\n   ${_doc}\n\n")
      endif()
    endif()
  endforeach()
endforeach()

set(SPHINX_CMAKE_VARIABLES_RST_DEST
    "${CMAKE_CURRENT_BINARY_DIR}/generated/cmake_variables.rst")
configure_file(
  "${PROJECT_SOURCE_DIR}/cmake/templates/cmake_variables.rst.in"
  ${SPHINX_CMAKE_VARIABLES_RST_DEST}
  @ONLY
)

set(toolchains_base_dir ${PROJECT_SOURCE_DIR}/cmake/toolchains/)
file(GLOB _toolchain_files ${toolchains_base_dir} ${toolchains_base_dir}*.cmake)

set(HPX_CMAKE_TOOLCHAINS_LIST_RST "")
set(HPX_CMAKE_TOOLCHAINS_RST "")
foreach(_toolchain ${_toolchain_files})
  get_filename_component(_toolchain_name ${_toolchain} NAME_WE)
  string(REGEX REPLACE "[ -]" "_" _toolchain_anchor "${_toolchain_name}")
  set(HPX_CMAKE_TOOLCHAINS_LIST_RST
   "${HPX_CMAKE_TOOLCHAINS_LIST_RST}* :ref:`${_toolchain_name} <${_toolchain_anchor}>`\n")
  string(REGEX REPLACE "." "-" _toolchain_underline ${_toolchain_name})
  set(HPX_CMAKE_TOOLCHAINS_RST
    "${HPX_CMAKE_TOOLCHAINS_RST}.. _${_toolchain_anchor}:\n\n${_toolchain_name}\n${_toolchain_underline}\n")
  set(HPX_CMAKE_TOOLCHAINS_RST "${HPX_CMAKE_TOOLCHAINS_RST}\n.. code-block:: cmake\n\n")
  file(STRINGS ${_toolchain} _toolchain_content)
  foreach(_line ${_toolchain_content})
    set(HPX_CMAKE_TOOLCHAINS_RST "${HPX_CMAKE_TOOLCHAINS_RST}   ${_line}\n")
  endforeach()
  set(HPX_CMAKE_TOOLCHAINS_RST "${HPX_CMAKE_TOOLCHAINS_RST}\n")
endforeach()

set(SPHINX_CMAKE_TOOLCHAINS_RST_DEST
    "${CMAKE_CURRENT_BINARY_DIR}/generated/cmake_toolchains.rst")
configure_file(
  "${PROJECT_SOURCE_DIR}/cmake/templates/cmake_toolchains.rst.in"
  ${SPHINX_CMAKE_TOOLCHAINS_RST_DEST}
  @ONLY
)

# Configure Sphinx conf.py
if(HPX_WITH_GIT_TAG)
  set(HPX_CONF_DOCUMENTATION_VERSION "${HPX_WITH_GIT_TAG}")
elseif(HPX_WITH_GIT_BRANCH)
  set(HPX_CONF_DOCUMENTATION_VERSION "${HPX_WITH_GIT_BRANCH}")
else()
  # HPX_VERSION is always available, use as fallback
  set(HPX_CONF_DOCUMENTATION_VERSION "${HPX_VERSION}")
endif()

set(SPHINX_CONF_PY_DEST
    "${CMAKE_CURRENT_BINARY_DIR}/sphinx/conf.py")
configure_file(
  "${PROJECT_SOURCE_DIR}/cmake/templates/conf.py.in"
  ${SPHINX_CONF_PY_DEST}
  @ONLY
)

# Copy the Sphinx source files to build directory
file(GLOB_RECURSE sphinx_source_files LIST_DIRECTORIES false RELATIVE
    "${CMAKE_CURRENT_SOURCE_DIR}/" "sphinx/*")

foreach(sphinx_source_file ${sphinx_source_files})
  add_custom_command(
    OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/${sphinx_source_file}"
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/${sphinx_source_file}"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
      "${CMAKE_CURRENT_SOURCE_DIR}/${sphinx_source_file}"
      "${CMAKE_CURRENT_BINARY_DIR}/${sphinx_source_file}")
endforeach()

foreach(sphinx_source_file ${sphinx_source_files})
  set(sphinx_source_files_build ${sphinx_source_files_build} "${CMAKE_CURRENT_BINARY_DIR}/${sphinx_source_file}")
endforeach()

# Create links to source files so that they are visible to sphinx
create_symbolic_link("${PROJECT_SOURCE_DIR}/src"
                     "${CMAKE_CURRENT_BINARY_DIR}/src")
create_symbolic_link("${PROJECT_SOURCE_DIR}/hpx"
                     "${CMAKE_CURRENT_BINARY_DIR}/hpx")
create_symbolic_link("${PROJECT_SOURCE_DIR}/examples"
                     "${CMAKE_CURRENT_BINARY_DIR}/examples")
create_symbolic_link("${PROJECT_SOURCE_DIR}/tests"
                     "${CMAKE_CURRENT_BINARY_DIR}/tests")

hpx_source_to_doxygen(hpx_autodoc
  DEPENDENCIES ${doxygen_dependencies})

set(SPHINX_DOCS_OUTPUT_DIR "${CMAKE_BINARY_DIR}/share/hpx-${HPX_VERSION}/docs/html")

add_custom_command(
  OUTPUT "${SPHINX_DOCS_OUTPUT_DIR}"
  DEPENDS "${sphinx_source_files_build}"
    "${CMAKE_CURRENT_BINARY_DIR}/hpx_autodoc/index.xml"
  COMMAND ${SPHINX_EXECUTABLE} -b html -n -d
    "${CMAKE_CURRENT_BINARY_DIR}/doctree"
    "${CMAKE_CURRENT_BINARY_DIR}/sphinx"
    "${SPHINX_DOCS_OUTPUT_DIR}")

add_custom_target(docs
  DEPENDS ${SPHINX_DOCS_OUTPUT_DIR})

add_custom_target(git_docs
  COMMAND "${CMAKE_COMMAND}"
    -DHPX_SOURCE_DIR:PATH=${CMAKE_SOURCE_DIR}
    -DHPX_BINARY_DIR:PATH=${CMAKE_BINARY_DIR}
    -DHPX_VERSION:STRING=${HPX_VERSION}
    -DHPX_WITH_GIT_BRANCH=${HPX_WITH_GIT_BRANCH}
    -DHPX_WITH_GIT_TAG=${HPX_WITH_GIT_TAG}
    -P "${CMAKE_SOURCE_DIR}/cmake/HPX_UpdateGitDocs.cmake")

set_target_properties(git_docs
    PROPERTIES FOLDER "Documentation/")

add_dependencies(git_docs docs)
