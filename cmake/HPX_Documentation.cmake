# Copyright (c) 2011 Bryce Lelbach
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(HPX_DOCUMENTATION_LOADED TRUE)

include(HPX_Include)

hpx_include(Message)

find_package(HPX_DocBook)
#find_package(HPX_BoostBook)
find_package(HPX_QuickBook)
find_package(HPX_Doxygen)
find_package(HPX_Xsltproc)

# issue a meaningful warning if part of the documentation toolchain is not available
if((NOT DOCBOOK_DTD_PATH_FOUND) OR (NOT DOCBOOK_XSL_PATH_FOUND))
  hpx_warn("documentation" "DocBook DTD or XSL is unavailable, documentation generation disabled. Set DOCBOOK_ROOT pointing to your DocBook installation directory.")
  set(HPX_BUILD_DOCUMENTATION OFF CACHE BOOL "True if the HPX documentation toolchain is available." FORCE)
#elseif((NOT BOOSTBOOK_DTD_PATH_FOUND) OR (NOT BOOSTBOOK_XSL_PATH_FOUND))
#  hpx_warn("documentation" "BoostBook DTD or XSL is unavailable, documentation generation disabled. Set BOOSTBOOK_ROOT pointing to your BoostBook installation directory.")
#  set(HPX_BUILD_DOCUMENTATION OFF CACHE BOOL "True if the HPX documentation toolchain is available." FORCE)
elseif(NOT QUICKBOOK_FOUND)
  hpx_warn("documentation" "QuickBook tool is unavailable, documentation generation disabled. Set QUICKBOOK_ROOT pointing to your QuickBook installation directory.")
  set(HPX_BUILD_DOCUMENTATION OFF CACHE BOOL "True if the HPX documentation toolchain is available." FORCE)
elseif(NOT XSLTPROC_FOUND)
  hpx_warn("documentation" "xsltproc tool is unavailable, documentation generation disabled. Set XSLTPROC_ROOT pointing to your xsltproc installation directory.")
  set(HPX_BUILD_DOCUMENTATION OFF CACHE BOOL "True if the HPX documentation toolchain is available." FORCE)
elseif(NOT DOXYGEN_FOUND)
  hpx_warn("documentation" "Doxygen tool is unavailable, API reference will be unavailable. Set DOXYGEN_ROOT pointing to your Doxygen installation directory.")
else()
  set(HPX_BUILD_DOCUMENTATION ON CACHE BOOL "True if the HPX documentation toolchain is available.")
endif()

if(NOT HPX_BUILD_DOCUMENTATION)
  # implement fallback macros for documentation toolchain

  macro(hpx_write_boostbook_catalog file)
    hpx_error("write_boostbook_catalog" "Documentation toolchain is unavailable.")
  endmacro()

  macro(hpx_quickbook_to_boostbook name)
    hpx_error("quickbook_to_boostbook" "Documentation toolchain is unavailable.")
  endmacro()

  macro(hpx_boostbook_to_docbook name)
    hpx_error("boostbook_to_docbook" "Documentation toolchain is unavailable.")
  endmacro()

  macro(hpx_quickbook_to_docbook name)
    hpx_error("quickbook_to_docbook" "Documentation toolchain is unavailable.")
  endmacro()

  macro(hpx_docbook_to_html name)
    hpx_error("docbook_to_html" "Documentation toolchain is unavailable.")
  endmacro()

  macro(hpx_source_to_doxygen name)
    hpx_error("source_to_doxygen" "Documentation toolchain is unavailable.")
  endmacro()

  macro(hpx_collect_doxygen name)
    hpx_error("collect_doxygen" "Documentation toolchain is unavailable.")
  endmacro()

  macro(hpx_doxygen_to_boostbook name)
    hpx_error("doxygen_to_boostbook" "Documentation toolchain is unavailable.")
  endmacro()

  macro(hpx_source_to_boostbook name)
    hpx_error("source_to_boostbook" "Documentation toolchain is unavailable.")
  endmacro()
else()
  set(BOOSTBOOK_DTD_PATH ${hpx_SOURCE_DIR}/external/boostbook/dtd/)
  set(BOOSTBOOK_XSL_PATH ${hpx_SOURCE_DIR}/external/boostbook/xsl/)

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
      "    rewritePrefix=\"file:///${DOCBOOK_DTD_PATH}/\"\n"
      "  />\n"
      "  <rewriteURI\n"
      "    uriStartString=\"http://docbook.sourceforge.net/release/xsl/current/\"\n"
      "    rewritePrefix=\"file:///${DOCBOOK_XSL_PATH}/\"\n"
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
      "    rewritePrefix=\"file:///${hpx_SOURCE_DIR}/external/boostbook/dtd/\"\n"
      "  />\n"
      "  <rewriteURI\n"
      "    uriStartString=\"http://www.boost.org/tools/boostbook/xsl/\"\n"
      "    rewritePrefix=\"file:///${hpx_SOURCE_DIR}/external/boostbook/xsl/\"\n"
      "  />\n"
      "</catalog>\n"
    )
  endmacro()

  # Quickbook -> BoostBook XML
  macro(hpx_quickbook_to_boostbook name)
    hpx_parse_arguments(${name} "SOURCE;DEPENDENCIES;QUICKBOOK_ARGS" "" ${ARGN})

    hpx_print_list("DEBUG"
        "quickbook_to_boostbook.${name}" "Quickbook arguments"
        ${name}_QUICKBOOK_ARGS)

    # If input is not a full path, it's in the current source directory.
    get_filename_component(input_path ${${name}_SOURCE} PATH)

    if(input_path STREQUAL "")
      set(input_path "${CMAKE_CURRENT_SOURCE_DIR}/${${name}_SOURCE}")
    else()
      set(input_path ${${name}_SOURCE})
    endif()

    set(svn_revision_option "")
    if(SVN_REVISION AND NOT ${SVN_REVISION} STREQUAL "")
      set(svn_revision_option "-D__hpx_svn_revision__=${SVN_REVISION}")
    endif()

    set(doxygen_option "")
    if(DOXYGEN_FOUND)
      set(doxygen_option "-D__hpx_doxygen__")
    endif()

    add_custom_command(OUTPUT ${name}.xml
      COMMAND ${QUICKBOOK_PROGRAM}
          "--output-file=${name}.xml"
          "${svn_revision_option}"
          "${doxygen_option}"
          "-D__hpx_source_dir__=${hpx_SOURCE_DIR}"
          "-D__hpx_binary_dir__=${CMAKE_BINARY_DIR}"
          "-D__hpx_docs_dir__=${CMAKE_CURRENT_BINARY_DIR}"
          ${${name}_QUICKBOOK_ARGS}
          ${input_path}
      COMMENT "Generating BoostBook XML file ${name}.xml from ${${name}_SOURCE}."
      DEPENDS ${${name}_SOURCE} ${${name}_DEPENDENCIES})
  endmacro()

  # BoostBook XML -> DocBook
  macro(hpx_boostbook_to_docbook name)
    hpx_parse_arguments(${name} "SOURCE;DEPENDENCIES;CATALOG;XSLTPROC_ARGS" "" ${ARGN})

    if(NOT BOOST_ROOT)
      set(BOOST_ROOT_FOR_DOCS ".")
    else()
      set(BOOST_ROOT_FOR_DOCS ${BOOST_ROOT})
    endif()

    if(WIN32)
      add_custom_command(OUTPUT ${name}.dbk
        COMMAND set XML_CATALOG_FILES=${${name}_CATALOG}
        COMMAND ${XSLTPROC_PROGRAM} ${${name}_XSLTPROC_ARGS}
                "--stringparam" "boost.graphics.root" "images/"
                "--stringparam" "admon.graphics.path" "images/"
                "--stringparam" "boost.root" "${BOOST_ROOT_FOR_DOCS}"
                "--stringparam" "html.stylesheet" "src/boostbook.css"
                "--xinclude" "-o" ${name}.dbk
                "--path" ${CMAKE_CURRENT_BINARY_DIR}
                ${BOOSTBOOK_XSL_PATH}/docbook.xsl ${${name}_SOURCE}
        COMMENT "Generating DocBook file ${name}.dbk from ${${name}_SOURCE}."
        DEPENDS ${${name}_SOURCE} ${${name}_DEPENDENCIES})
    else()
      add_custom_command(OUTPUT ${name}.dbk
        COMMAND "XML_CATALOG_FILES=${${name}_CATALOG}" ${XSLTPROC_PROGRAM}
                ${${name}_XSLTPROC_ARGS}
                "--stringparam" "boost.graphics.root" "images/"
                "--stringparam" "admon.graphics.path" "images/"
                "--stringparam" "boost.root" "${BOOST_ROOT_FOR_DOCS}"
                "--stringparam" "html.stylesheet" "src/boostbook.css"
                "--xinclude" "-o" ${name}.dbk
                "--path" ${CMAKE_CURRENT_BINARY_DIR}
                ${BOOSTBOOK_XSL_PATH}/docbook.xsl ${${name}_SOURCE}
        COMMENT "Generating DocBook file ${name}.dbk from ${${name}_SOURCE}."
        DEPENDS ${${name}_SOURCE} ${${name}_DEPENDENCIES})
    endif()
  endmacro()

  # DocBook -> HTML
  macro(hpx_docbook_to_html name)
    hpx_parse_arguments(${name} "SOURCE;DEPENDENCIES;CATALOG;XSLTPROC_ARGS" "" ${ARGN})

    if(NOT BOOST_ROOT)
      set(BOOST_ROOT_FOR_DOCS ".")
    else()
      set(BOOST_ROOT_FOR_DOCS ${BOOST_ROOT})
    endif()

    if(WIN32)
      set(DOCS_OUTPUT_DIR "file:///${CMAKE_BINARY_DIR}/${CMAKE_BUILD_TYPE}/share/hpx-${HPX_VERSION}/docs/html/")
      add_custom_command(OUTPUT ${name}_HTML.manifest
        COMMAND set XML_CATALOG_FILES=${${name}_CATALOG}
        COMMAND ${XSLTPROC_PROGRAM} ${${name}_XSLTPROC_ARGS}
                "--stringparam" "boost.graphics.root" "images/"
                "--stringparam" "admon.graphics.path" "images/"
                "--stringparam" "boost.root" "${BOOST_ROOT_FOR_DOCS}"
                "--stringparam" "html.stylesheet" "src/boostbook.css"
                "--stringparam" "manifest" "${CMAKE_CURRENT_BINARY_DIR}/${name}_HTML.manifest"
                "--xinclude"
                "-o" "${DOCS_OUTPUT_DIR}"
                "--path" ${CMAKE_CURRENT_BINARY_DIR}
                ${BOOSTBOOK_XSL_PATH}/html.xsl ${${name}_SOURCE}
        COMMENT "Generating HTML from ${${name}_SOURCE}."
        DEPENDS ${${name}_SOURCE} ${${name}_DEPENDENCIES})
    else()
      set(DOCS_OUTPUT_DIR "file:///${CMAKE_BINARY_DIR}/share/hpx-${HPX_VERSION}/docs/html/")
      add_custom_command(OUTPUT ${name}_HTML.manifest
        COMMAND "XML_CATALOG_FILES=${${name}_CATALOG}" ${XSLTPROC_PROGRAM}
                ${${name}_XSLTPROC_ARGS}
                "--stringparam" "boost.graphics.root" "images/"
                "--stringparam" "admon.graphics.path" "images/"
                "--stringparam" "boost.root" "${BOOST_ROOT_FOR_DOCS}"
                "--stringparam" "html.stylesheet" "src/boostbook.css"
                "--stringparam" "manifest" "${CMAKE_CURRENT_BINARY_DIR}/${name}_HTML.manifest"
                "--xinclude"
                "-o" "${DOCS_OUTPUT_DIR}"
                "--path" ${CMAKE_CURRENT_BINARY_DIR}
                ${BOOSTBOOK_XSL_PATH}/html.xsl ${${name}_SOURCE}
        COMMENT "Generating HTML from ${${name}_SOURCE}."
        DEPENDS ${${name}_SOURCE} ${${name}_DEPENDENCIES})
    endif()
  endmacro()

  # Quickbook -> BoostBook XML -> DocBook -> HTML
  macro(hpx_quickbook_to_html name)
    hpx_parse_arguments(${name}
        "SOURCE;DEPENDENCIES;CATALOG;XSLTPROC_ARGS;TARGET;QUICKBOOK_ARGS"
        "" ${ARGN})

    hpx_print_list("DEBUG"
        "quickbook_to_html.${name}" "Documentation dependencies"
        ${name}_DEPENDENCIES)

    hpx_quickbook_to_boostbook(${name}
      SOURCE ${${name}_SOURCE}
      DEPENDENCIES ${${name}_DEPENDENCIES}
      QUICKBOOK_ARGS ${${name}_QUICKBOOK_ARGS})

    hpx_boostbook_to_docbook(${name}
      SOURCE ${name}.xml
      CATALOG ${${name}_CATALOG}
      XSLTPROC_ARGS ${${name}_XSLTPROC_ARGS})

    hpx_docbook_to_html(${name}
      SOURCE ${name}.dbk
      CATALOG ${${name}_CATALOG}
      XSLTPROC_ARGS ${${name}_XSLTPROC_ARGS})

    if(${name}_TARGET)
      add_custom_target(${${name}_TARGET} DEPENDS ${name}_HTML.manifest
        DEPENDENCIES ${${name}_DEPENDENCIES})
    endif()
  endmacro()

  # C++ Source -> Doxygen XML
  macro(hpx_source_to_doxygen name)
    hpx_parse_arguments(${name} "DEPENDENCIES;DOXYGEN_ARGS" "" ${ARGN})

    hpx_print_list("DEBUG"
        "hpx_source_to_doxygen.${name}" "Doxygen dependencies"
        ${name}_DEPENDENCIES)

    add_custom_command(OUTPUT ${name}/index.xml
      COMMAND ${DOXYGEN_PROGRAM} ${${name}_DOXYGEN_ARGS}
              ${CMAKE_CURRENT_BINARY_DIR}/${name}.doxy
      COMMENT "Generating Doxygen."
      DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${name}.doxy ${${name}_DEPENDENCIES})
  endmacro()

  # Collect chunked Doxygen XML
  macro(hpx_collect_doxygen name)
    hpx_parse_arguments(${name} "SOURCE;DEPENDENCIES;CATALOG;XSLTPROC_ARGS" "" ${ARGN})

    if(WIN32)
      add_custom_command(OUTPUT ${name}.doxygen.xml
        COMMAND set XML_CATALOG_FILES=${${name}_CATALOG}
        COMMAND ${XSLTPROC_PROGRAM} ${${name}_XSLTPROC_ARGS}
#                "--stringparam" "doxygen.xml.path" ${CMAKE_CURRENT_BINARY_DIR}/${name}
                "--xinclude" "-o" ${name}.doxygen.xml
                "--path" ${CMAKE_CURRENT_BINARY_DIR}
                ${BOOSTBOOK_XSL_PATH}/doxygen/collect.xsl ${${name}_SOURCE}
        COMMENT "Collecting Doxygen XML files."
        DEPENDS ${${name}_SOURCE} ${${name}_DEPENDENCIES})
    else()
      add_custom_command(OUTPUT ${name}.doxygen.xml
        COMMAND "XML_CATALOG_FILES=${${name}_CATALOG}" ${XSLTPROC_PROGRAM}
                ${${name}_XSLTPROC_ARGS}
#                "--stringparam" "doxygen.xml.path" ${CMAKE_CURRENT_BINARY_DIR}/${name}
                "--xinclude" "-o" ${name}.doxygen.xml
                "--path" ${CMAKE_CURRENT_BINARY_DIR}
                ${BOOSTBOOK_XSL_PATH}/doxygen/collect.xsl ${${name}_SOURCE}
        COMMENT "Collecting Doxygen XML files."
        DEPENDS ${${name}_SOURCE} ${${name}_DEPENDENCIES})
    endif()
  endmacro()

  # Doxygen XML -> BoostBook XML
  macro(hpx_doxygen_to_boostbook name)
    hpx_parse_arguments(${name} "SOURCE;DEPENDENCIES;CATALOG;XSLTPROC_ARGS" "" ${ARGN})

    if(WIN32)
      add_custom_command(OUTPUT ${name}.xml
        COMMAND set XML_CATALOG_FILES=${${name}_CATALOG}
        COMMAND ${XSLTPROC_PROGRAM} ${${name}_XSLTPROC_ARGS}
                "--stringparam" "boost.doxygen.header.stripped_prefix" ${hpx_SOURCE_DIR}
                "--stringparam" "boost.doxygen.header.added_prefix" "file:///${CMAKE_INSTALL_PREFIX}/include/"
                "--xinclude" "-o" ${name}.xml
                "--path" ${CMAKE_CURRENT_BINARY_DIR}
                ${BOOSTBOOK_XSL_PATH}/doxygen/doxygen2boostbook.xsl
                ${${name}_SOURCE}
        COMMENT "Generating BoostBook XML file ${name}.xml from ${${name}_SOURCE}."
        DEPENDS ${${name}_SOURCE} ${${name}_DEPENDENCIES})
    else()
      add_custom_command(OUTPUT ${name}.xml
        COMMAND "XML_CATALOG_FILES=${${name}_CATALOG}" ${XSLTPROC_PROGRAM}
                ${${name}_XSLTPROC_ARGS}
                "--stringparam" "boost.doxygen.header.stripped_prefix" ${hpx_SOURCE_DIR}
                "--stringparam" "boost.doxygen.header.added_prefix" "file://${CMAKE_INSTALL_PREFIX}/include/"
                "--xinclude" "-o" ${name}.xml
                "--path" ${CMAKE_CURRENT_BINARY_DIR}
                ${BOOSTBOOK_XSL_PATH}/doxygen/doxygen2boostbook.xsl
                ${${name}_SOURCE}
        COMMENT "Generating BoostBook XML file ${name}.xml from ${${name}_SOURCE}."
        DEPENDS ${${name}_SOURCE} ${${name}_DEPENDENCIES})
    endif()
  endmacro()

  # C++ Source -> Doxygen XML -> BoostBook XML
  macro(hpx_source_to_boostbook name)
    hpx_parse_arguments(${name}
        "DEPENDENCIES;CATALOG;XSLTPROC_ARGS;TARGET;DOXYGEN_ARGS"
        "" ${ARGN})

    hpx_source_to_doxygen(${name}
      DEPENDENCIES ${${name}_DEPENDENCIES}
      DOXYGEN_ARGS ${${name}_DOXYGEN_ARGS})

    hpx_collect_doxygen(${name}
      SOURCE ${name}/index.xml
      CATALOG ${${name}_CATALOG}
      XSLTPROC_ARGS ${${name}_XSLTPROC_ARGS})

    hpx_doxygen_to_boostbook(${name}
      SOURCE ${name}.doxygen.xml
      CATALOG ${${name}_CATALOG}
      XSLTPROC_ARGS ${${name}_XSLTPROC_ARGS})

    if(${name}_TARGET)
      add_custom_target(${${name}_TARGET} DEPENDS ${name}.xml
        DEPENDENCIES ${${name}_DEPENDENCIES})
    endif()
  endmacro()
endif()

