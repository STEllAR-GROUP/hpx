# Copyright (c) 2011 Bryce Lelbach
# Copyright (c) 2012-2013 Hartmut Kaiser
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

if(HPX_WITH_DOCUMENTATION)
  find_package(Doxygen)
  find_package(Sphinx)
  find_package(Breathe)

  if(NOT SPHINX_FOUND)
    hpx_error("Sphinx is unavailable, sphinx documentation generation disabled. Set SPHINX_ROOT to your sphinx-build installation directory.")
    set(HPX_WITH_DOCUMENTATION OFF)
  elseif(NOT BREATHE_FOUND)
    hpx_error("Breathe is unavailable, sphinx documentation generation disabled. Set BREATHE_APIDOC_ROOT to your breathe-apidoc installation directory.")
    set(HPX_WITH_DOCUMENTATION OFF)
  elseif(NOT DOXYGEN_FOUND)
    hpx_error("Doxygen tool is unavailable, sphinx documentation generation disabled. Add the doxygen executable to your path or set the DOXYGEN_EXECUTABLE variable manually.")
    set(HPX_WITH_DOCUMENTATION OFF)
  endif()
endif()

# C++ Source -> Doxygen XML
function(hpx_source_to_doxygen name)
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
endfunction()
