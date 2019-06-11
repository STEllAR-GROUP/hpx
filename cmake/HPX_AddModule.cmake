# Copyright (c) 2019 Auriane Reverdell
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

function(add_hpx_module name)
  # Retrieve arguments
  set(options DEPRECATION_WARNINGS COMPATIBILITY_HEADERS)
  set(one_value_args SOURCE_ROOT HEADER_ROOT COMPAT_HEADER_ROOT)
  set(multi_value_args SOURCES HEADERS COMPAT_HEADERS DEPENDENCIES CMAKE_SUBDIRS)
  cmake_parse_arguments(${name} "${options}" "${one_value_args}" "${multi_value_args}" ${ARGN})

  if(NOT ${name}_SOURCE_ROOT)
      set(${name}_SOURCE_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/src")
  endif()
  hpx_debug("Add module ${name}: ${name}_SOURCE_ROOT: ${${name}_SOURCE_ROOT}")

  if(NOT ${name}_HEADER_ROOT)
      set(${name}_HEADER_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/include")
  endif()
  hpx_debug("Add module ${name}: ${name}_HEADER_ROOT: ${${name}_HEADER_ROOT}")

  if(HPX_${name_upper}_WITH_COMPATIBILITY_HEADERS)
    if(NOT ${name}_COMPAT_HEADER_ROOT)
        set(${name}_COMPAT_HEADER_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/include_compatibility")
    endif()
    hpx_debug("Add module ${name}: ${name}_COMPAT_HEADER_ROOT: ${${name}_COMPAT_HEADER_ROOT}")
  endif()

  string(TOUPPER ${name} name_upper)

  hpx_option(HPX_${name_upper}_WITH_TESTS
    BOOL
    "Build HPX ${name} module tests. (default: ON)"
    ON ADVANCED
    CATEGORY "Modules")
  
  if(${name}_DEPRECATION_WARNINGS)
    hpx_option(HPX_${name_upper}_WITH_DEPRECATION_WARNINGS
      BOOL
      "Enable warnings for deprecated facilities. (default: ${HPX_WITH_DEPRECATION_WARNINGS})"
      ${HPX_WITH_DEPRECATION_WARNINGS} ADVANCED
      CATEGORY "Modules")
    
    if(HPX_${name}_WITH_DEPRECATION_WARNINGS)
      hpx_add_config_define_namespace(
          DEFINE HPX_${name_upper}_HAVE_DEPRECATION_WARNINGS
          NAMESPACE ${name_upper})
    endif()
  endif()
  
  if(${name}_COMPATIBILITY_HEADERS)
    # Added in 1.4.0
    hpx_option(HPX_${name_upper}_WITH_COMPATIBILITY_HEADERS
      BOOL
      "Enable compatibility headers for old headers"
      ON ADVANCED
      CATEGORY "Modules")
    
    if(HPX_${name_upper}_WITH_COMPATIBILITY_HEADERS)
      hpx_add_config_define_namespace(
          DEFINE HPX_${name_upper}_HAVE_COMPATIBILITY_HEADERS
          NAMESPACE ${name_upper})
    endif()
  endif()

  set(sources ${${name}_SOURCES})
  set(tmp_headers ${${name}_HEADERS})
  set(compat_headers ${${name}_COMPAT_HEADERS})

  # Add a global include file that include all module headers
  FILE(WRITE ${${name}_HEADER_ROOT}/hpx/${name}/${name}_module.hpp
      "//  Copyright (c) 2019 The STE||AR GROUP\n"
      "//\n"
      "//  Distributed under the Boost Software License, Version 1.0. (See accompanying\n"
      "//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)\n\n"
  )
  foreach(header_file ${tmp_headers})
    FILE(APPEND ${${name}_HEADER_ROOT}/hpx/${name}/${name}_module.hpp
      "#include <${header_file}>\n"
    )
    list(APPEND headers "${${name}_HEADER_ROOT}/${header_file}")
  endforeach(header_file)

  foreach(header_file ${headers})
      hpx_debug(${header_file})
  endforeach(header_file)

  message(STATUS "${name}: Configuring")
  
  add_library(hpx_${name} STATIC ${sources} ${headers} ${compat_headers})
  
  target_link_libraries(hpx_${name} ${${name}_DEPENDENCIES})
  target_include_directories(hpx_${name} PUBLIC
    $<BUILD_INTERFACE:${${name}_HEADER_ROOT}>
    $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}>)
  
  if(HPX_${name_upper}_WITH_COMPATIBILITY_HEADERS)
    target_include_directories(hpx_${name} PUBLIC
      $<BUILD_INTERFACE:${${name}_COMPAT_HEADER_ROOT}>)
  endif()
  
  target_compile_definitions(hpx_${name} PRIVATE
    $<$<CONFIG:Debug>:DEBUG>
    $<$<CONFIG:Debug>:_DEBUG>
    HPX_MODULE_EXPORTS
  )
  
  add_hpx_source_group(
    NAME hpx_{name}
    ROOT ${${name}_HEADER_ROOT}/hpx
    CLASS "Header Files"
    TARGETS ${headers})
  add_hpx_source_group(
    NAME hpx_{name}
    ROOT ${${name}_SOURCE_ROOT}/hpx
    CLASS "Source Files"
    TARGETS ${sources})
  
  if(HPX_${name_upper}_WITH_COMPATIBILITY_HEADERS)
    add_hpx_source_group(
      NAME hpx_${name}
      ROOT ${${name}_COMPAT_HEADER_ROOT}/hpx
      CLASS "Header Files"
      TARGETS ${compat_headers})
  endif()
  
  set_target_properties(hpx_${name} PROPERTIES
    FOLDER "Core/Modules")
  
  install(TARGETS hpx_${name} EXPORT HPXTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    COMPONENT ${name}
  )
  hpx_export_targets(hpx_${name})
  
  install(
    DIRECTORY include/hpx
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    COMPONENT ${name})
  
  if(HPX_${name_upper}_WITH_COMPATIBILITY_HEADERS)
    install(
      DIRECTORY include_compatibility/hpx
      DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
      COMPONENT ${name})
  endif()
  
  write_config_defines_file(
    NAMESPACE ${name_upper}
    FILENAME "${CMAKE_BINARY_DIR}/hpx/${name}/config/defines.hpp")
  
  write_config_defines_file(
    NAMESPACE ${name_upper}
    FILENAME "${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/hpx/${name}/config/defines.hpp")
  
  foreach(dir ${${name}_CMAKE_SUBDIRS})
    add_subdirectory(${dir})
  endforeach(dir)
  
  message(STATUS "${name}: Configuring done")

endfunction()
