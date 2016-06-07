# Copyright (c) 2011 Bryce Lelbach
# Copyright (c) 2012-2013 Hartmut Kaiser
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

if(HPX_WITH_DOCUMENTATION)
  find_package(DocBook)
  find_package(BoostQuickBook)
  find_package(Doxygen)
  find_package(XSLTPROC)
  find_package(BoostAutoIndex)
  find_package(FOP)


  # issue a meaningful warning if part of the documentation toolchain is not available
  if(NOT DOCBOOK_FOUND)
    hpx_error("DocBook DTD or XSL is unavailable, documentation generation disabled. Set DOCBOOK_ROOT to your DocBook installation directory.")
    set(HPX_WITH_DOCUMENTATION OFF)
  elseif(NOT BOOSTQUICKBOOK_FOUND)
    hpx_error("Boost QuickBook tool is unavailable, documentation generation disabled. Set BOOSTQUICKBOOK_ROOT or BOOST_ROOT to your Boost QuickBook installation directory.")
    set(HPX_WITH_DOCUMENTATION OFF)
  elseif(NOT XSLTPROC_FOUND)
    hpx_error("xsltproc tool is unavailable, documentation generation disabled. Add the xsltproc executable to your path or set XSLTPROC_ROOT.")
    set(HPX_WITH_DOCUMENTATION OFF)
  elseif(NOT DOXYGEN_FOUND)
    hpx_error("Doxygen tool is unavailable, API reference will be unavailable. Add the doxygen executable to your path or set the DOXYGEN_EXECUTABLE variable manually.")
    set(HPX_WITH_DOCUMENTATION OFF)
  elseif(NOT BOOSTAUTOINDEX_FOUND)
    hpx_error("Boost auto_index tool is unavailable, index generation will be disabled. Set BOOSTAUTOINDEX_ROOT to your Boost auto_index installation directory.")
    set(HPX_WITH_DOCUMENTATION OFF)
  elseif(NOT FOP_FOUND)
    hpx_info("FOP is unavailable, PDF generation will be disabled. Set FOP_ROOT to your FOP installation directory.")
  endif()
endif()


set(BOOSTBOOK_DTD_PATH "${PROJECT_SOURCE_DIR}/external/boostbook/dtd/")
set(BOOSTBOOK_XSL_PATH "${PROJECT_SOURCE_DIR}/external/boostbook/xsl/")

# Generate catalog file for XSLT processing
macro(hpx_write_boostbook_catalog file)
  file(WRITE ${file}
    "<?xml version=\"1.0\"?>\n"
    "<!--\n"
    " Copyright (c) 2011-2012 Bryce Adelstein-Lelbach\n"
    "\n"
    " Distributed under the Boost Software License, Version 1.0. (See accompanying\n"
    " file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)\n"
    "-->\n"
    "<!DOCTYPE catalog\n"
    "  PUBLIC \"-//OASIS/DTD Entity Resolution XML Catalog V1.0//EN\"\n"
    "  \"http://www.oasis-open.org/committees/entity/release/1.0/catalog.dtd\">\n"
    "<catalog xmlns=\"urn:oasis:names:tc:entity:xmlns:xml:catalog\">\n"
    "  <rewriteURI\n"
    "    uriStartString=\"http://www.oasis-open.org/docbook/xml/4.2/\"\n"
    "    rewritePrefix=\"file:///${DOCBOOK_DTD}/\"\n"
    "  />\n"
    "  <rewriteURI\n"
    "    uriStartString=\"http://docbook.sourceforge.net/release/xsl/current/\"\n"
    "    rewritePrefix=\"file:///${DOCBOOK_XSL}/\"\n"
    "  />\n"
    #"   <rewriteURI\n"
    #"    uriStartString=\"http://www.boost.org/tools/boostbook/dtd/\"\n"
    #"    rewritePrefix=\"file:///${BOOSTBOOK_DTD_PATH}/\"\n"
    #"  />\n"
    #"  <rewriteURI\n"
    #"    uriStartString=\"http://www.boost.org/tools/boostbook/xsl/\"\n"
    #"    rewritePrefix=\"file:///${BOOSTBOOK_XSL_PATH}/\"\n"
    #"  />\n"
    "   <rewriteURI\n"
    "    uriStartString=\"http://www.boost.org/tools/boostbook/dtd/\"\n"
    "    rewritePrefix=\"file:///${PROJECT_SOURCE_DIR}/external/boostbook/dtd/\"\n"
    "  />\n"
    "  <rewriteURI\n"
    "    uriStartString=\"http://www.boost.org/tools/boostbook/xsl/\"\n"
    "    rewritePrefix=\"file:///${PROJECT_SOURCE_DIR}/external/boostbook/xsl/\"\n"
    "  />\n"
    "</catalog>\n"
  )
endmacro()

# Quickbook -> BoostBook XML
macro(hpx_quickbook_to_boostbook name)
  set(options NODOXYGEN)
  set(one_value_args SOURCE)
  set(multi_value_args DEPENDENCIES QUICKBOOK_ARGS)
  cmake_parse_arguments(${name} "${options}" "${one_value_args}" "${multi_value_args}" ${ARGN})

  # If input is not a full path, it's in the current source directory.
  get_filename_component(input_path ${${name}_SOURCE} PATH)

  if(input_path STREQUAL "")
    set(input_path "${CMAKE_CURRENT_SOURCE_DIR}/${${name}_SOURCE}")
  else()
    set(input_path "${${name}_SOURCE}")
  endif()

  set(git_commit_option "")
  if(HPX_WITH_GIT_COMMIT AND NOT ${HPX_WITH_GIT_COMMIT} STREQUAL "")
    set(git_commit_option "-D__hpx_git_commit__=${HPX_WITH_GIT_COMMIT}")
  endif()

  set(doxygen_option "")
  if(DOXYGEN_FOUND AND NOT ${name}_NODOXYGEN)
    set(doxygen_option "-D__hpx_doxygen__")
  endif()

  set(doc_source_dir "'''./code'''")
  add_custom_command(OUTPUT ${name}.xml
    COMMAND "${BOOSTQUICKBOOK_EXECUTABLE}"
        "--output-file=${name}.xml"
        "${git_commit_option}"
        "${doxygen_option}"
        "-D__hpx_source_dir__=${doc_source_dir}"
        "-D__hpx_binary_dir__=${CMAKE_BINARY_DIR}"
        "-D__hpx_docs_dir__=${CMAKE_CURRENT_BINARY_DIR}"
        ${${name}_QUICKBOOK_ARGS}
        "${input_path}"
    COMMENT "Generating BoostBook XML file ${name}.xml from ${${name}_SOURCE}."
    DEPENDS "${${name}_SOURCE}" ${${name}_DEPENDENCIES}
    VERBATIM)
endmacro()

# BoostBook XML -> DocBook
macro(hpx_boostbook_to_docbook name)
  set(options)
  set(one_value_args SOURCE )
  set(multi_value_args DEPENDENCIES CATALOG XSLTPROC_ARGS)
  cmake_parse_arguments(${name} "${options}" "${one_value_args}" "${multi_value_args}" ${ARGN})

  if(NOT BOOST_ROOT)
    set(BOOST_ROOT_FOR_DOCS ".")
  else()
    set(BOOST_ROOT_FOR_DOCS "${BOOST_ROOT}")
  endif()

  xsltproc(
    OUTPUT "${name}.dbk"
    STYLESHEET "${BOOSTBOOK_XSL_PATH}/docbook.xsl"
    INPUT "${${name}_SOURCE}"
    XINCLUDE
    PARAMETERS
      boost.graphics.root=images/
      admon.graphics.path=images/
      callout.graphics.path=images/
      boost.root=${BOOST_ROOT_FOR_DOCS}
      html.stylesheet=src/boostbook.css
      ${${name}_XSLTPROC_ARGS}
    CATALOG ${${name}_CATALOG}
    DEPENDS ${${name}_DEPENDENCIES}
  )
endmacro()

# DocBook -> XSL-FO
macro(hpx_docbook_to_xslfo name)
  set(options)
  set(one_value_args SOURCE )
  set(multi_value_args DEPENDENCIES CATALOG XSLTPROC_ARGS)
  cmake_parse_arguments(${name} "${options}" "${one_value_args}" "${multi_value_args}" ${ARGN})

  if(NOT BOOST_ROOT)
    set(BOOST_ROOT_FOR_DOCS ".")
  else()
    set(BOOST_ROOT_FOR_DOCS "${BOOST_ROOT}")
  endif()

  xsltproc(
    OUTPUT "${name}.fo"
    STYLESHEET "${BOOSTBOOK_XSL_PATH}/fo.xsl"
    INPUT "${${name}_SOURCE}"
    XINCLUDE
    PARAMETERS
      paper.type=USLetter
      admon.graphics.extension=.png
      img.src.path=${PROJECT_SOURCE_DIR}/docs/html/
      boost.graphics.root=${PROJECT_SOURCE_DIR}/docs/html/images/
      admon.graphics.path=${PROJECT_SOURCE_DIR}/docs/html/images/
      callout.graphics.path=${PROJECT_SOURCE_DIR}/docs/html/images/
      ${${name}_XSLTPROC_ARGS}
    CATALOG ${${name}_CATALOG}
    DEPENDS ${${name}_DEPENDENCIES}
  )
endmacro()

# XSL-FO -> PDF
macro(hpx_xslfo_to_pdf name)
  set(options)
  set(one_value_args SOURCE )
  set(multi_value_args DEPENDENCIES FOP_ARGS)
  cmake_parse_arguments(${name} "${options}" "${one_value_args}" "${multi_value_args}" ${ARGN})

  set(DOCS_OUTPUT_DIR "/${CMAKE_BINARY_DIR}/share/hpx-${HPX_VERSION}/docs/")

  add_custom_command(OUTPUT "${name}.pdf"
    COMMAND "${FOP_EXECUTABLE}" ${${name}_FOP_ARGS}
            "${${name}_SOURCE}" "${DOCS_OUTPUT_DIR}/${name}.pdf"
    COMMENT "Generating PDF file ${name}.pdf from ${${name}_SOURCE}."
    DEPENDS "${${name}_SOURCE}" ${${name}_DEPENDENCIES}
    VERBATIM)
endmacro()

# DocBook -> HTML
macro(hpx_docbook_to_html name)
  set(options)
  set(one_value_args SOURCE SINGLEPAGE)
  set(multi_value_args DEPENDENCIES CATALOG XSLTPROC_ARGS)
  cmake_parse_arguments(${name} "${options}" "${one_value_args}" "${multi_value_args}" ${ARGN})

  if(NOT BOOST_ROOT)
    set(BOOST_ROOT_FOR_DOCS ".")
  else()
    set(BOOST_ROOT_FOR_DOCS "${BOOST_ROOT}")
  endif()

  set(DOCS_OUTPUT_DIR "file:///${CMAKE_BINARY_DIR}/share/hpx-${HPX_VERSION}/docs/html")

  if(${${name}_SINGLEPAGE})
    set(main_xsl_script "html-single.xsl")
    set(main_xsl_script_output "${DOCS_OUTPUT_DIR}/${name}.html")
    set(main_xsl_script_manifest "${name}_singlepage_HTML.manifest")
  else()
    set(main_xsl_script "html.xsl")
    set(main_xsl_script_output "${DOCS_OUTPUT_DIR}/")
    set(main_xsl_script_manifest "${name}_HTML.manifest")
  endif()
  xsltproc(
    TARGET ${main_xsl_script_manifest}
    OUTPUT ${main_xsl_script_output}
    STYLESHEET ${BOOSTBOOK_XSL_PATH}/${main_xsl_script}
    INPUT "${${name}_SOURCE}"
    XINCLUDE
    PARAMETERS
      boost.graphics.root=images/
      admon.graphics.path=images/
      callout.graphics.path=images/
      boost.root=${BOOST_ROOT_FOR_DOCS}
      html.stylesheet=src/boostbook.css
      manifest=${CMAKE_CURRENT_BINARY_DIR}/${main_xsl_script_manifest}
      ${${name}_XSLTPROC_ARGS}
    CATALOG ${${name}_CATALOG}
    DEPENDS ${${name}_DEPENDENCIES}
  )
endmacro()

# Quickbook -> BoostBook XML -> DocBook -> HTML, AutoIndex (if available)
macro(hpx_quickbook_to_html name)
  set(options)
  set(one_value_args SOURCE INDEX TARGET SINGLEPAGE)
  set(multi_value_args DEPENDENCIES CATALOG XSLTPROC_ARGS QUICKBOOK_ARGS AUTOINDEX_ARGS)
  cmake_parse_arguments(${name} "${options}" "${one_value_args}" "${multi_value_args}" ${ARGN})

  hpx_quickbook_to_boostbook(${name}
    SOURCE "${${name}_SOURCE}"
    DEPENDENCIES ${${name}_DEPENDENCIES}
    QUICKBOOK_ARGS ${${name}_QUICKBOOK_ARGS})

  hpx_boostbook_to_docbook(${name}
    SOURCE "${name}.xml"
    CATALOG ${${name}_CATALOG}
    XSLTPROC_ARGS ${${name}_XSLTPROC_ARGS})

  set(docbook_source "${name}.dbk")
  if(BOOSTAUTOINDEX_FOUND)
    hpx_generate_auto_index(${name}
      INDEX "${${name}_INDEX}"
      SOURCE "${name}.dbk"
      AUTOINDEX_ARGS ${${name}_AUTOINDEX_ARGS})
    set(docbook_source "${name}_auto_index.dbk")
  endif()

  if(${${name}_SINGLEPAGE})
    hpx_docbook_to_html(${name}
      SOURCE ${docbook_source}
      CATALOG ${${name}_CATALOG}
      XSLTPROC_ARGS ${${name}_XSLTPROC_ARGS}
      SINGLEPAGE ON)
  endif()

  hpx_docbook_to_html(${name}
    SOURCE ${docbook_source}
    CATALOG ${${name}_CATALOG}
    XSLTPROC_ARGS ${${name}_XSLTPROC_ARGS}
    SINGLEPAGE OFF)

  if(${name}_TARGET)
    add_custom_target(${${name}_TARGET}
      DEPENDS ${name}_HTML.manifest
      DEPENDENCIES ${${name}_DEPENDENCIES})

    if(${name}_SINGLEPAGE)
      add_custom_target(${${name}_TARGET}
        DEPENDS ${name}_singlepage_HTML.manifest
        DEPENDENCIES ${${name}_DEPENDENCIES})
    endif()
  endif()
endmacro()

# C++ Source -> Doxygen XML
macro(hpx_source_to_doxygen name)
  set(options)
  set(one_value_args)
  set(multi_value_args DEPENDENCIES DOXYGEN_ARGS)
  cmake_parse_arguments(${name} "${options}" "${one_value_args}" "${multi_value_args}" ${ARGN})

  add_custom_command(OUTPUT "${name}/index.xml"
    COMMAND "${DOXYGEN_EXECUTABLE}" ${${name}_DOXYGEN_ARGS}
            "${CMAKE_CURRENT_BINARY_DIR}/${name}.doxy"
    COMMENT "Generating Doxygen."
    DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/${name}.doxy" ${${name}_DEPENDENCIES}
    VERBATIM)
endmacro()

# Collect chunked Doxygen XML
macro(hpx_collect_doxygen name)
  set(options)
  set(one_value_args SOURCE)
  set(multi_value_args DEPENDENCIES CATALOG XSLTPROC_ARGS)
  cmake_parse_arguments(${name} "${options}" "${one_value_args}" "${multi_value_args}" ${ARGN})

  xsltproc(
    OUTPUT "${name}.doxygen.xml"
    STYLESHEET "${BOOSTBOOK_XSL_PATH}/doxygen/collect.xsl"
    INPUT "${${name}_SOURCE}"
    XINCLUDE
    PARAMETERS
      doxygen.xml.path=${CMAKE_CURRENT_BINARY_DIR}/${name}
      ${${name}_XSLTPROC_ARGS}
    CATALOG ${${name}_CATALOG}
    DEPENDS ${${name}_DEPENDENCIES}
  )
endmacro()

# Doxygen XML -> BoostBook XML
macro(hpx_doxygen_to_boostbook name)
  set(options)
  set(one_value_args SOURCE)
  set(multi_value_args DEPENDENCIES CATALOG XSLTPROC_ARGS)
  cmake_parse_arguments(${name} "${options}" "${one_value_args}" "${multi_value_args}" ${ARGN})

  xsltproc(
    OUTPUT "${name}.xml"
    STYLESHEET "${BOOSTBOOK_XSL_PATH}/doxygen/doxygen2boostbook.xsl"
    INPUT "${${name}_SOURCE}"
    XINCLUDE
    PARAMETERS
      boost.doxygen.header.prefix=hpx
      ${${name}_XSLTPROC_ARGS}
    CATALOG ${${name}_CATALOG}
    DEPENDS ${${name}_DEPENDENCIES}
  )
endmacro()

# C++ Source -> Doxygen XML -> BoostBook XML
macro(hpx_source_to_boostbook name)
  set(options)
  set(one_value_args TARGET)
  set(multi_value_args DEPENDENCIES CATALOG XSLTPROC_ARGS DOXYGEN_ARGS)
  cmake_parse_arguments(${name} "${options}" "${one_value_args}" "${multi_value_args}" ${ARGN})

  hpx_source_to_doxygen(${name}
    DEPENDENCIES ${${name}_DEPENDENCIES}
    DOXYGEN_ARGS ${${name}_DOXYGEN_ARGS})

  hpx_collect_doxygen(${name}
    SOURCE "${name}/index.xml"
    CATALOG ${${name}_CATALOG}
    XSLTPROC_ARGS ${${name}_XSLTPROC_ARGS})

  hpx_doxygen_to_boostbook(${name}
    SOURCE "${name}.doxygen.xml"
    CATALOG ${${name}_CATALOG}
    XSLTPROC_ARGS ${${name}_XSLTPROC_ARGS})

  if(${name}_TARGET)
    add_custom_target(${${name}_TARGET} DEPENDS "${name}.xml"
      DEPENDENCIES ${${name}_DEPENDENCIES})
  endif()
endmacro()

# Generate auto_index reference for docs
macro(hpx_generate_auto_index name)
  set(options)
  set(one_value_args SOURCE INDEX)
  set(multi_value_args AUTOINDEX_ARGS)
  cmake_parse_arguments(${name} "${options}" "${one_value_args}" "${multi_value_args}" ${ARGN})

  # If input is not a full path, it's in the current source directory.
  get_filename_component(input_path ${${name}_INDEX} PATH)

  if(input_path STREQUAL "")
    set(input_path "${CMAKE_CURRENT_SOURCE_DIR}/${${name}_INDEX}")
  else()
    set(input_path "${${name}_INDEX}")
  endif()

  add_custom_command(OUTPUT "${name}_auto_index.dbk"
    COMMAND "${BOOSTAUTOINDEX_EXECUTABLE}" ${${name}_AUTOINDEX_ARGS}
            "--script=${input_path}"
            "--in=${${name}_SOURCE}"
            "--out=${name}_auto_index.dbk"
    COMMENT "Generating auto index."
    DEPENDS "${${name}_SOURCE}" "${${name}_INDEX}"
    VERBATIM)
endmacro()

