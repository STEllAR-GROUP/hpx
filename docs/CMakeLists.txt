# Copyright (c) 2012 Bryce Adelstein-Lelbach
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

# Add all files here which should be passed to doxygen
set(doxygen_dependencies
    "${PROJECT_SOURCE_DIR}/components/component_storage/include/hpx/components/component_storage/migrate_from_storage.hpp"
    "${PROJECT_SOURCE_DIR}/components/component_storage/include/hpx/components/component_storage/migrate_to_storage.hpp"
    "${PROJECT_SOURCE_DIR}/hpx/runtime_fwd.hpp"
    "${PROJECT_SOURCE_DIR}/hpx/runtime/basename_registration_fwd.hpp"
    "${PROJECT_SOURCE_DIR}/hpx/runtime/find_here.hpp"
    "${PROJECT_SOURCE_DIR}/hpx/runtime/find_localities.hpp"
    "${PROJECT_SOURCE_DIR}/hpx/runtime/get_colocation_id.hpp"
    "${PROJECT_SOURCE_DIR}/hpx/runtime/get_ptr.hpp"
    "${PROJECT_SOURCE_DIR}/hpx/runtime/get_locality_name.hpp"
    "${PROJECT_SOURCE_DIR}/hpx/runtime/set_parcel_write_handler.hpp"
    "${PROJECT_SOURCE_DIR}/hpx/runtime/trigger_lco.hpp"
    "${PROJECT_SOURCE_DIR}/hpx/runtime/components/binpacking_distribution_policy.hpp"
    "${PROJECT_SOURCE_DIR}/hpx/runtime/components/colocating_distribution_policy.hpp"
    "${PROJECT_SOURCE_DIR}/hpx/runtime/components/component_factory.hpp"
    "${PROJECT_SOURCE_DIR}/hpx/runtime/components/copy_component.hpp"
    "${PROJECT_SOURCE_DIR}/hpx/runtime/components/default_distribution_policy.hpp"
    "${PROJECT_SOURCE_DIR}/hpx/runtime/components/migrate_component.hpp"
    "${PROJECT_SOURCE_DIR}/hpx/runtime/components/new.hpp"
    "${PROJECT_SOURCE_DIR}/hpx/runtime/naming/unmanaged.hpp"
)

# Definitions for autodoc.doxy.in
foreach(doxygen_input ${doxygen_dependencies})
  set(doxygen_inputs "${doxygen_inputs} ${doxygen_input}")
endforeach()

set(doxygen_output_file "${CMAKE_CURRENT_BINARY_DIR}/hpx_autodoc")
set(doxygen_output_dir "${CMAKE_CURRENT_BINARY_DIR}/doxygen")

set(doxygen_definition_list
    "DOXYGEN:=1"
    "BOOST_SYSTEM_NOEXCEPT="
    "HPX_EXCEPTION_EXPORT="
    "HPX_CORE_EXPORT="
    "HPX_PARALLELISM_EXPORT="
    "HPX_FULL_EXPORT="
    "HPX_EXPORT="
    "HPX_ALWAYS_EXPORT="
    "extern="
    "HPX_FORCEINLINE="
    "HPX_CONCEPT_REQUIRES_(...)="
    "HPX_STATIC_CONSTEXPR=static constexpr"
    "HPX_HOST_DEVICE="
)

foreach(doxygen_predef ${doxygen_definition_list})
  set(doxygen_definitions "${doxygen_definitions} \"${doxygen_predef}\"")
endforeach()

hpx_info("Creating Doxyfile: ${CMAKE_CURRENT_BINARY_DIR}/hpx_autodoc.doxy")
configure_file(
  "${PROJECT_SOURCE_DIR}/cmake/templates/autodoc.doxy.in"
  "${CMAKE_CURRENT_BINARY_DIR}/hpx_autodoc.doxy" @ONLY
)

# Generate rst files with CMake variables and toolchains
set(HPX_CMAKE_OPTIONS_RST "")
foreach(_cat ${HPX_OPTION_CATEGORIES})
  string(REPLACE " " "_" _cat_link "${_cat}")
  set(HPX_CMAKE_OPTIONS_RST
      "${HPX_CMAKE_OPTIONS_RST}* :ref:`${_cat} options <${_cat_link}>`\n"
  )
endforeach()
set(HPX_CMAKE_OPTIONS_RST "${HPX_CMAKE_OPTIONS_RST}\n")

get_cmake_property(_variableNames CACHE_VARIABLES)
foreach(_cat ${HPX_OPTION_CATEGORIES})
  string(REPLACE " " "_" _cat_anchor "${_cat}")
  set(_cat_title "${_cat} options")
  string(REGEX REPLACE "." "-" _cat_title_underline "${_cat_title}")
  set(HPX_CMAKE_OPTIONS_RST
      "${HPX_CMAKE_OPTIONS_RST}.. _${_cat_anchor}:\n\n${_cat_title}\n${_cat_title_underline}\n\n"
  )
  foreach(_variableName ${_variableNames})
    if(${_variableName}Category)
      if(_cat STREQUAL ${_variableName}Category)
        string(REPLACE " " "_" _variableName_anchor "${_variableName}")

        get_property(
          _doc
          CACHE "${_variableName}"
          PROPERTY HELPSTRING
        )
        get_property(
          _type
          CACHE "${_variableName}"
          PROPERTY TYPE
        )
        if(NOT _type STREQUAL "UNINITIALIZED")
          set(_variableName "${_variableName}:${_type}")
        endif()
        set(HPX_CMAKE_OPTIONS_RST
            "${HPX_CMAKE_OPTIONS_RST}* :option:`${_variableName}`\n"
        )
      endif()
    endif()
  endforeach()
  set(HPX_CMAKE_OPTIONS_RST "${HPX_CMAKE_OPTIONS_RST}\n")

  foreach(_variableName ${_variableNames})
    if(${_variableName}Category)
      if(_cat STREQUAL ${_variableName}Category)
        string(REPLACE " " "_" _variableName_anchor "${_variableName}")

        get_property(
          _doc
          CACHE "${_variableName}"
          PROPERTY HELPSTRING
        )
        get_property(
          _type
          CACHE "${_variableName}"
          PROPERTY TYPE
        )
        if(NOT _type STREQUAL "UNINITIALIZED")
          set(_variableName "${_variableName}:${_type}")
        endif()
        set(HPX_CMAKE_OPTIONS_RST
            "${HPX_CMAKE_OPTIONS_RST}.. option:: ${_variableName}\n\n   ${_doc}\n\n"
        )
      endif()
    endif()
  endforeach()
endforeach()

set(SPHINX_CMAKE_VARIABLES_RST_DEST
    "${CMAKE_CURRENT_BINARY_DIR}/generated/cmake_variables.rst"
)
configure_file(
  "${PROJECT_SOURCE_DIR}/cmake/templates/cmake_variables.rst.in"
  ${SPHINX_CMAKE_VARIABLES_RST_DEST} @ONLY
)

set(toolchains_base_dir ${PROJECT_SOURCE_DIR}/cmake/toolchains/)
file(GLOB _toolchain_files ${toolchains_base_dir} ${toolchains_base_dir}*.cmake)

set(HPX_CMAKE_TOOLCHAINS_LIST_RST "")
set(HPX_CMAKE_TOOLCHAINS_RST "")
foreach(_toolchain ${_toolchain_files})
  get_filename_component(_toolchain_name ${_toolchain} NAME_WE)
  string(REGEX REPLACE "[ -]" "_" _toolchain_anchor "${_toolchain_name}")
  set(HPX_CMAKE_TOOLCHAINS_LIST_RST
      "${HPX_CMAKE_TOOLCHAINS_LIST_RST}* :ref:`${_toolchain_name} <${_toolchain_anchor}>`\n"
  )
  string(REGEX REPLACE "." "-" _toolchain_underline ${_toolchain_name})
  set(HPX_CMAKE_TOOLCHAINS_RST
      "${HPX_CMAKE_TOOLCHAINS_RST}.. _${_toolchain_anchor}:\n\n${_toolchain_name}\n${_toolchain_underline}\n"
  )
  set(HPX_CMAKE_TOOLCHAINS_RST
      "${HPX_CMAKE_TOOLCHAINS_RST}\n.. code-block:: cmake\n\n"
  )
  file(STRINGS ${_toolchain} _toolchain_content)
  foreach(_line ${_toolchain_content})
    set(HPX_CMAKE_TOOLCHAINS_RST "${HPX_CMAKE_TOOLCHAINS_RST}   ${_line}\n")
  endforeach()
  set(HPX_CMAKE_TOOLCHAINS_RST "${HPX_CMAKE_TOOLCHAINS_RST}\n")
endforeach()

set(SPHINX_CMAKE_TOOLCHAINS_RST_DEST
    "${CMAKE_CURRENT_BINARY_DIR}/generated/cmake_toolchains.rst"
)
configure_file(
  "${PROJECT_SOURCE_DIR}/cmake/templates/cmake_toolchains.rst.in"
  ${SPHINX_CMAKE_TOOLCHAINS_RST_DEST} @ONLY
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

set(SPHINX_CONF_PY_DEST "${CMAKE_CURRENT_BINARY_DIR}/sphinx/conf.py")
configure_file(
  "${PROJECT_SOURCE_DIR}/cmake/templates/conf.py.in" ${SPHINX_CONF_PY_DEST}
  @ONLY
)

# Copy the Sphinx source files to build directory
file(
  GLOB_RECURSE sphinx_source_files
  LIST_DIRECTORIES false
  RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}/"
  "sphinx/*"
)

# exclude api.rst...
list(FILTER sphinx_source_files EXCLUDE REGEX "sphinx/api.rst")

foreach(sphinx_source_file ${sphinx_source_files})
  add_custom_command(
    OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/${sphinx_source_file}"
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/${sphinx_source_file}"
    COMMAND
      ${CMAKE_COMMAND} -E copy_if_different
      "${CMAKE_CURRENT_SOURCE_DIR}/${sphinx_source_file}"
      "${CMAKE_CURRENT_BINARY_DIR}/${sphinx_source_file}"
  )
endforeach()

foreach(sphinx_source_file ${sphinx_source_files})
  set(sphinx_source_files_build
      ${sphinx_source_files_build}
      "${CMAKE_CURRENT_BINARY_DIR}/${sphinx_source_file}"
  )
endforeach()

# Copy the Sphinx source files of the libs directory to build directory
file(
  GLOB_RECURSE sphinx_libs_source_files
  LIST_DIRECTORIES false
  RELATIVE "${PROJECT_SOURCE_DIR}/libs"
  "${PROJECT_SOURCE_DIR}/libs/*/*/docs/*"
)

list(
  APPEND
  sphinx_libs_source_files
  "index.rst"
  "overview.rst"
  "core/modules.rst"
  "full/modules.rst"
  "parallelism/modules.rst"
)

foreach(lib ${HPX_LIBS})
  list(APPEND sphinx_libs_source_files "${lib}/modules.rst")
endforeach()

foreach(sphinx_source_file ${sphinx_libs_source_files})
  add_custom_command(
    OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/sphinx/libs/${sphinx_source_file}"
    DEPENDS "${PROJECT_SOURCE_DIR}/libs/${sphinx_source_file}"
    COMMAND
      ${CMAKE_COMMAND} -E copy_if_different
      "${PROJECT_SOURCE_DIR}/libs/${sphinx_source_file}"
      "${CMAKE_CURRENT_BINARY_DIR}/sphinx/libs/${sphinx_source_file}"
  )
endforeach()

foreach(sphinx_source_file ${sphinx_libs_source_files})
  set(sphinx_source_files_build
      ${sphinx_source_files_build}
      "${CMAKE_CURRENT_BINARY_DIR}/sphinx/libs/${sphinx_source_file}"
  )
endforeach()

# Create links to source files so that they are visible to sphinx
create_symbolic_link(
  "${PROJECT_SOURCE_DIR}/src" "${CMAKE_CURRENT_BINARY_DIR}/src"
)
create_symbolic_link(
  "${PROJECT_SOURCE_DIR}/hpx" "${CMAKE_CURRENT_BINARY_DIR}/hpx"
)
create_symbolic_link(
  "${PROJECT_SOURCE_DIR}/examples" "${CMAKE_CURRENT_BINARY_DIR}/examples"
)
create_symbolic_link(
  "${PROJECT_SOURCE_DIR}/tests" "${CMAKE_CURRENT_BINARY_DIR}/tests"
)
create_symbolic_link(
  "${PROJECT_SOURCE_DIR}/libs" "${CMAKE_CURRENT_BINARY_DIR}/libs"
)

hpx_source_to_doxygen(hpx_autodoc DEPENDENCIES ${doxygen_dependencies})

add_custom_target(docs)

set(SPHINX_DOCS_HTML_OUTPUT_FILE "index.html")
set(SPHINX_DOCS_SINGLEHTML_OUTPUT_FILE "index.html")
set(SPHINX_DOCS_PDF_OUTPUT_FILE "latex/HPX.pdf")
set(SPHINX_DOCS_MAN_OUTPUT_FILE "hpx.1")

foreach(output_format ${HPX_WITH_DOCUMENTATION_OUTPUT_FORMATS})
  string(TOUPPER ${output_format} output_format_upper_case)
  set(SPHINX_DOCS_OUTPUT_DIR
      "${PROJECT_BINARY_DIR}/share/hpx/docs/${output_format}"
  )
  set(SPHINX_DOCS_OUTPUT_FILE
      "${SPHINX_DOCS_OUTPUT_DIR}/${SPHINX_DOCS_${output_format_upper_case}_OUTPUT_FILE}"
  )

  # NOTE: The PDF does not build without errors but still produces a reasonable
  # PDF. If there are real errors this will ignore them. Fixing the errors is at
  # the moment not worth the effort.
  if(${output_format} STREQUAL "latexpdf")
    set(SPHINX_DOCS_BUILD_COMMAND
        ${SPHINX_EXECUTABLE} -M ${output_format}
        "${CMAKE_CURRENT_BINARY_DIR}/sphinx" "${SPHINX_DOCS_OUTPUT_DIR}" || (
                                                                            exit
                                                                            0)
    )
  else()
    set(SPHINX_DOCS_BUILD_COMMAND
        ${SPHINX_EXECUTABLE} -b ${output_format} -n -d
        "${CMAKE_CURRENT_BINARY_DIR}/doctree"
        "${CMAKE_CURRENT_BINARY_DIR}/sphinx" "${SPHINX_DOCS_OUTPUT_DIR}"
    )
  endif()

  add_custom_command(
    OUTPUT "${SPHINX_DOCS_OUTPUT_FILE}"
    DEPENDS "${sphinx_source_files_build}"
            "${CMAKE_CURRENT_BINARY_DIR}/hpx_autodoc/index.xml"
    COMMAND ${SPHINX_DOCS_BUILD_COMMAND}
  )

  add_custom_target(
    docs-${output_format} ALL DEPENDS ${SPHINX_DOCS_OUTPUT_FILE}
  )

  add_dependencies(docs docs-${output_format})
endforeach()

add_custom_target(
  git_docs
  COMMAND
    "${CMAKE_COMMAND}" -DHPX_BINARY_DIR:PATH=${PROJECT_BINARY_DIR}
    -DHPX_VERSION:STRING=${HPX_VERSION}
    -DHPX_WITH_GIT_BRANCH=${HPX_WITH_GIT_BRANCH}
    -DHPX_WITH_GIT_TAG=${HPX_WITH_GIT_TAG}
    -DHPX_WITH_DOCUMENTATION_OUTPUT_FORMATS="${HPX_WITH_DOCUMENTATION_OUTPUT_FORMATS}"
    -P "${PROJECT_SOURCE_DIR}/cmake/HPX_UpdateGitDocs.cmake"
)

set_target_properties(git_docs PROPERTIES FOLDER "Documentation/")

add_dependencies(git_docs docs)
