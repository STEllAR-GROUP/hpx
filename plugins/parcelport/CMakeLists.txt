# Copyright (c) 2007-2017 Hartmut Kaiser
# Copyright (c) 2014-2015 Thomas Heller
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

# We explicitly disable ibverbs and ipc for now as they don't work currently

set(parcelport_plugins)

if(HPX_WITH_NETWORKING)
  set(parcelport_plugins ${parcelport_plugins}
    libfabric
    verbs
    mpi
    tcp)
endif()

set(HPX_STATIC_PARCELPORT_PLUGINS "" CACHE INTERNAL "" FORCE)
set(HPX_STATIC_PARCELPORT_PLUGINS_SOURCES "" CACHE INTERNAL "" FORCE)
set(HPX_STATIC_PARCELPORT_PLUGINS_PROPERTIES "" CACHE INTERNAL "" FORCE)

macro(add_parcelport name)
  set(name_short ${name})
  set(name "parcelport_${name}")
  set(options STATIC)
  set(one_value_args FOLDER)
  set(multi_value_args SOURCES HEADERS DEPENDENCIES COMPILE_FLAGS LINK_FLAGS)
  cmake_parse_arguments(${name} "${options}" "${one_value_args}" "${multi_value_args}" ${ARGN})

  if(${${name}_STATIC})
    hpx_debug("adding static parcelport: ${name}")

    add_hpx_source_group(
      NAME hpx
      CLASS "Header Files"
      ROOT "${PROJECT_SOURCE_DIR}/hpx"
      TARGETS ${${name}_HEADERS})
    add_hpx_source_group(
      NAME hpx
      CLASS "Source Files"
      ROOT "${PROJECT_SOURCE_DIR}"
      TARGETS ${${name}_SOURCES})

    set(HPX_STATIC_PARCELPORT_PLUGINS
      ${HPX_STATIC_PARCELPORT_PLUGINS} ${name}
      CACHE INTERNAL "" FORCE)

    set(HPX_STATIC_PARCELPORT_PLUGINS_HEADERS
      ${HPX_STATIC_PARCELPORT_PLUGINS_HEADERS}
      ${${name}_HEADERS})
    list(SORT HPX_STATIC_PARCELPORT_PLUGINS_HEADERS)
    list(REMOVE_DUPLICATES HPX_STATIC_PARCELPORT_PLUGINS_HEADERS)
    set(HPX_STATIC_PARCELPORT_PLUGINS_HEADERS
      ${HPX_STATIC_PARCELPORT_PLUGINS_HEADERS}
      CACHE INTERNAL "" FORCE)

    set(HPX_STATIC_PARCELPORT_PLUGINS_SOURCES
      ${HPX_STATIC_PARCELPORT_PLUGINS_SOURCES}
      ${${name}_SOURCES})
    list(SORT HPX_STATIC_PARCELPORT_PLUGINS_SOURCES)
    list(REMOVE_DUPLICATES HPX_STATIC_PARCELPORT_PLUGINS_SOURCES)
    set(HPX_STATIC_PARCELPORT_PLUGINS_SOURCES
      ${HPX_STATIC_PARCELPORT_PLUGINS_SOURCES}
      CACHE INTERNAL "" FORCE)

    hpx_libraries(${${name}_DEPENDENCIES})

  else()
    hpx_debug("adding parcelport plugin: ${name}")
    add_hpx_library(${name}
        PLUGIN
        SOURCES ${${name}_SOURCES}
        HEADERS ${${name}_HEADERS}
        FOLDER ${${name}_FOLDER}
        COMPILE_FLAGS ${${name}_COMPILE_FLAGS}
        LINK_FLAGS ${${name}_LINK_FLAGS}
        DEPENDENCIES ${${name}_DEPENDENCIES})

    add_hpx_pseudo_dependencies(plugins.parcelport.${name_short} ${name}_lib)
    add_hpx_pseudo_dependencies(core plugins.parcelport.${name_short})
  endif()

endmacro()

foreach(type ${parcelport_plugins})
  add_hpx_pseudo_target(plugins.parcelport.${type})
  add_subdirectory(${type})
  add_hpx_pseudo_dependencies(plugins.parcelport plugins.parcelport.${type})
endforeach()

macro(add_static_parcelports)
  if(HPX_WITH_NETWORKING)
    add_parcelport_tcp_module()
    add_parcelport_mpi_module()
    add_parcelport_verbs_module()
    add_parcelport_libfabric_module()
  endif()
endmacro()

macro(add_parcelport_modules)
  if(HPX_WITH_NETWORKING)
    hpx_debug("creating static_parcelports.hpp: " ${HPX_STATIC_PARCELPORT_PLUGINS})

    set(_parcelport_export)
    set(_parcelport_init)
    foreach(parcelport ${HPX_STATIC_PARCELPORT_PLUGINS})
      set(_parcelport_export
        "${_parcelport_export}HPX_EXPORT hpx::plugins::parcelport_factory_base *parcelport_${parcelport}_factory_init(std::vector<hpx::plugins::parcelport_factory_base *>& factories);\n")
      set(_parcelport_init
        "${_parcelport_init}        parcelport_${parcelport}_factory_init(factories);\n")
    endforeach()
  endif()

  configure_file("${PROJECT_SOURCE_DIR}/cmake/templates/static_parcelports.hpp.in"
                 "${CMAKE_BINARY_DIR}/hpx/runtime/parcelset/static_parcelports.hpp"
                 @ONLY)
endmacro()

#------------------------------------------------------------------------------
# Write parcelport_defines.hpp
#------------------------------------------------------------------------------
# write config to build dir
write_config_defines_file(
  TEMPLATE ""
  NAMESPACE "parcelport"
  FILENAME "${CMAKE_BINARY_DIR}/hpx/config/parcelport_defines.hpp")

# write config to install dir
write_config_defines_file(
  TEMPLATE ""
  NAMESPACE "parcelport"
  FILENAME
  "${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/hpx/config/parcelport_defines.hpp")

