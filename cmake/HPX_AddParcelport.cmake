# Copyright (c) 2019 Ste||ar Group
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

macro(add_parcelport name)

  set(name_short ${name})
  set(name "parcelport_${name}")
  set(options STATIC)
  set(one_value_args FOLDER)
  set(multi_value_args SOURCES HEADERS DEPENDENCIES INCLUDE_DIRS COMPILE_FLAGS LINK_FLAGS)
  cmake_parse_arguments(${name} "${options}" "${one_value_args}" "${multi_value_args}" ${ARGN})

  hpx_debug("adding static parcelport: ${name}")

  # Add source file for visual studio
  add_hpx_source_group(
    NAME hpx
    CLASS "Header Files"
    ROOT ${PROJECT_SOURCE_DIR}/hpx
    TARGETS ${${name}_HEADERS})
  add_hpx_source_group(
    NAME hpx
    CLASS "Source Files"
    ROOT ${PROJECT_SOURCE_DIR}
    TARGETS ${${name}_SOURCES})

  set(HPX_STATIC_PARCELPORT_PLUGINS
    ${HPX_STATIC_PARCELPORT_PLUGINS} ${name}
    CACHE INTERNAL "" FORCE)

  add_library(${name} STATIC ${${name}_SOURCES} ${${name}_HEADERS})
  target_link_libraries(${name} PUBLIC ${${name}_DEPENDENCIES})
  # TODO : put some generator expressions
  target_include_directories(${name} PUBLIC ${${name}_INCLUDE_DIRS})
  target_link_libraries(${name} PRIVATE hpx_internal_flags)
  target_compile_options(${name} PUBLIC ${${name}_COMPILE_FLAGS})
  set_target_properties(${name} PROPERTIES
    FOLDER "${${name}_FOLDER}"
    LINK_FLAGS "${${name}_LINK_FLAGS}"
    POSITION_INDEPENDENT_CODE ON)

  if ({name}_EXPORT)
    get_target_property(_link_libraries ${name} INTERFACE_LINK_LIBRARIES)
    hpx_export_targets(${_link_libraries})
  endif()

  add_hpx_pseudo_dependencies(plugins.parcelport.${name_short} ${name})
  add_hpx_pseudo_dependencies(core plugins.parcelport.${name_short})

endmacro()
