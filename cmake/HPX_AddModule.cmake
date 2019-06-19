# Copyright (c) 2019 Auriane Reverdell
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

function(add_hpx_module name)
  # Retrieve arguments
  set(options DEPRECATION_WARNINGS)
  set(one_value_args COMPATIBILITY_HEADERS GLOBAL_HEADER_GEN)
  set(multi_value_args SOURCES HEADERS COMPAT_HEADERS DEPENDENCIES CMAKE_SUBDIRS)
  cmake_parse_arguments(${name} "${options}" "${one_value_args}" "${multi_value_args}" ${ARGN})

  project(HPX.${name} CXX)

  include(HPX_Message)
  include(HPX_Option)
  
  hpx_info("  ${name}")

  # Compatibility header off is not specified
  if (NOT ${name}_COMPATIBILITY_HEADERS)
      set(${name}_COMPATIBILITY_HEADERS OFF)
  endif()

  if ("${${name}_GLOBAL_HEADER_GEN}" STREQUAL "")
      set(${name}_GLOBAL_HEADER_GEN ON)
  endif()

  # Main directories of the module
  set(SOURCE_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/src")
  hpx_debug("Add module ${name}: SOURCE_ROOT: ${SOURCE_ROOT}")
  set(HEADER_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/include")
  hpx_debug("Add module ${name}: HEADER_ROOT: ${HEADER_ROOT}")
  set(COMPAT_HEADER_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/include_compatibility")
  hpx_debug("Add module ${name}: COMPAT_HEADER_ROOT: ${COMPAT_HEADER_ROOT}")

  string(TOUPPER ${name} name_upper)

  # HPX options

  hpx_option(HPX_${name_upper}_WITH_TESTS
    BOOL
    "Build HPX ${name} module tests. (default: ${HPX_WITH_TESTS})"
    ${HPX_WITH_TESTS} ADVANCED
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

  # Write full path for the sources files
  include(HPX_CMakeUtils)
  prepend(sources ${SOURCE_ROOT} ${${name}_SOURCES})
  prepend(headers ${HEADER_ROOT} ${${name}_HEADERS})
  if(HPX_${name_upper}_WITH_COMPATIBILITY_HEADERS)
    prepend(compat_headers ${COMPAT_HEADER_ROOT} ${${name}_COMPAT_HEADERS})
  endif()

  # This header generation is disabled for config module specific generated
  # headers are included
  if (${name}_GLOBAL_HEADER_GEN)
    # Add a global include file that include all module headers
    FILE(WRITE ${CMAKE_BINARY_DIR}/hpx/${name}.hpp
        "//  Copyright (c) 2019 The STE||AR GROUP\n"
        "//\n"
        "//  Distributed under the Boost Software License, Version 1.0. (See accompanying\n"
        "//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)\n\n"
        "#ifndef HPX_${name_upper}_HPP\n"
        "#define HPX_${name_upper}_HPP\n\n"
    )
    foreach(header_file ${${name}_HEADERS})
      FILE(APPEND ${CMAKE_BINARY_DIR}/hpx/${name}.hpp
        "#include <${header_file}>\n"
      )
    endforeach(header_file)
    FILE(APPEND ${CMAKE_BINARY_DIR}/hpx/${name}.hpp
      "\n#endif"
    )
  endif()

  foreach(header_file ${headers})
      hpx_debug(${header_file})
  endforeach(header_file)

  add_library(hpx_${name} STATIC ${sources} ${headers} ${compat_headers})
  
  target_link_libraries(hpx_${name} ${${name}_DEPENDENCIES})
  target_include_directories(hpx_${name} PUBLIC
    $<BUILD_INTERFACE:${HEADER_ROOT}>
    $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}>)
  
  if(HPX_${name_upper}_WITH_COMPATIBILITY_HEADERS)
    target_include_directories(hpx_${name} PUBLIC
      $<BUILD_INTERFACE:${COMPAT_HEADER_ROOT}>)
  endif()
  
  target_compile_definitions(hpx_${name} PRIVATE
    $<$<CONFIG:Debug>:DEBUG>
    $<$<CONFIG:Debug>:_DEBUG>
    HPX_MODULE_EXPORTS
  )
  
  add_hpx_source_group(
    NAME hpx_{name}
    ROOT ${HEADER_ROOT}/hpx
    CLASS "Header Files"
    TARGETS ${headers})
  add_hpx_source_group(
    NAME hpx_{name}
    ROOT ${SOURCE_ROOT}/hpx
    CLASS "Source Files"
    TARGETS ${sources})
  if(HPX_${name_upper}_WITH_COMPATIBILITY_HEADERS)
    add_hpx_source_group(
      NAME hpx_${name}
      ROOT ${COMPAT_HEADER_ROOT}/hpx
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
  
endfunction()
