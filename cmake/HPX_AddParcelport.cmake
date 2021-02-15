# Copyright (c) 2019 Ste||ar Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

function(add_parcelport name)

  set(options STATIC EXPORT)
  set(one_value_args FOLDER)
  set(multi_value_args SOURCES HEADERS DEPENDENCIES INCLUDE_DIRS COMPILE_FLAGS
                       LINK_FLAGS
  )
  cmake_parse_arguments(
    ${name} "${options}" "${one_value_args}" "${multi_value_args}" ${ARGN}
  )

  hpx_debug("adding static parcelport: ${name}")
  set(parcelport_name "parcelport_${name}")

  # Add source file for visual studio
  add_hpx_source_group(
    NAME hpx
    CLASS "Header Files"
    ROOT ${PROJECT_SOURCE_DIR}/hpx
    TARGETS ${${name}_HEADERS}
  )
  add_hpx_source_group(
    NAME hpx
    CLASS "Source Files"
    ROOT ${PROJECT_SOURCE_DIR}
    TARGETS ${${name}_SOURCES}
  )

  if(${name}_STATIC)
    set(HPX_STATIC_PARCELPORT_PLUGINS
        ${HPX_STATIC_PARCELPORT_PLUGINS} ${parcelport_name}
        CACHE INTERNAL "" FORCE
    )
  endif()

  add_library(${parcelport_name} STATIC ${${name}_SOURCES} ${${name}_HEADERS})

  target_link_libraries(${parcelport_name} PUBLIC ${${name}_DEPENDENCIES})
  target_include_directories(${parcelport_name} PUBLIC ${${name}_INCLUDE_DIRS})
  target_link_libraries(
    ${parcelport_name}
    PUBLIC hpx_public_flags
    PRIVATE hpx_private_flags
  )
  target_compile_options(${parcelport_name} PUBLIC ${${name}_COMPILE_FLAGS})
  set_target_properties(
    ${parcelport_name}
    PROPERTIES FOLDER "${${name}_FOLDER}"
               LINK_FLAGS "${${name}_LINK_FLAGS}"
               POSITION_INDEPENDENT_CODE ON
  )

  if(HPX_WITH_UNITY_BUILD)
    set_target_properties(${parcelport_name} PROPERTIES UNITY_BUILD ON)
  endif()

  if(HPX_WITH_PRECOMPILED_HEADERS)
    target_precompile_headers(
      ${parcelport_name} REUSE_FROM hpx_precompiled_headers
    )
  endif()

  if(${name}_EXPORT)
    get_target_property(
      _link_libraries ${parcelport_name} INTERFACE_LINK_LIBRARIES
    )
    hpx_export_targets(${_link_libraries})
  endif()

  target_compile_definitions(${parcelport_name} PRIVATE HPX_EXPORTS)

  add_hpx_pseudo_dependencies(plugins.parcelport.${name} ${parcelport_name})
  add_hpx_pseudo_dependencies(core plugins.parcelport.${name})

endfunction()
