# Copyright (c) 2019 Auriane Reverdell
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

function(add_hpx_module name)
  # Retrieve arguments
  set(options DEPRECATION_WARNINGS)
  set(one_value_args COMPATIBILITY_HEADERS GLOBAL_HEADER_GEN FORCE_LINKING_GEN INSTALL_BINARIES)
  set(multi_value_args SOURCES HEADERS COMPAT_HEADERS DEPENDENCIES CMAKE_SUBDIRS)
  cmake_parse_arguments(${name} "${options}" "${one_value_args}" "${multi_value_args}" ${ARGN})

  project(HPX.${name} CXX)

  include(HPX_Message)
  include(HPX_Option)

  hpx_info("  ${name}")

  # Global headers should be always generated except if explicitly disabled
  if ("${${name}_GLOBAL_HEADER_GEN}" STREQUAL "")
      set(${name}_GLOBAL_HEADER_GEN ON)
  endif()

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

  # Main directories of the module
  set(SOURCE_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/src")
  set(HEADER_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/include")

  hpx_debug("Add module ${name}: SOURCE_ROOT: ${SOURCE_ROOT}")
  hpx_debug("Add module ${name}: HEADER_ROOT: ${HEADER_ROOT}")

  if(HPX_${name_upper}_WITH_COMPATIBILITY_HEADERS)
    set(COMPAT_HEADER_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/include_compatibility")
    hpx_debug("Add module ${name}: COMPAT_HEADER_ROOT: ${COMPAT_HEADER_ROOT}")
  endif()

  # Write full path for the sources files
  include(HPX_CMakeUtils)
  prepend(sources ${SOURCE_ROOT} ${${name}_SOURCES})
  prepend(headers ${HEADER_ROOT} ${${name}_HEADERS})
  if(HPX_${name_upper}_WITH_COMPATIBILITY_HEADERS)
    prepend(compat_headers ${COMPAT_HEADER_ROOT} ${${name}_COMPAT_HEADERS})
  endif()

  set(copyright
    "//  Copyright (c) 2019 The STE||AR GROUP\n"
    "//\n"
    "//  Distributed under the Boost Software License, Version 1.0. (See accompanying\n"
    "//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)\n"
    "\n"
  )

  # This header generation is disabled for config module specific generated
  # headers are included
  if (${name}_GLOBAL_HEADER_GEN)
    set(global_header "${CMAKE_CURRENT_BINARY_DIR}/include/hpx/${name}.hpp")
    # Add a global include file that include all module headers
    FILE(WRITE ${global_header}
        ${copyright}
        "#ifndef HPX_${name_upper}_HPP\n"
        "#define HPX_${name_upper}_HPP\n\n"
    )
    foreach(header_file ${${name}_HEADERS})
      FILE(APPEND ${global_header}
        "#include <${header_file}>\n"
      )
    endforeach(header_file)
    FILE(APPEND ${global_header}
      "\n#endif\n"
    )
  endif()

  if(${name}_FORCE_LINKING_GEN)
      # Add a header to force linking of modules on Windows
      set(force_linking_header "${CMAKE_CURRENT_BINARY_DIR}/include/hpx/${name}/force_linking.hpp")
      FILE(WRITE ${force_linking_header}
          ${copyright}
          "#if !defined(HPX_${name_upper}_FORCE_LINKING_HPP)\n"
          "#define HPX_${name_upper}_FORCE_LINKING_HPP\n"
          "\n"
          "namespace hpx { namespace ${name}\n"
          "{\n"
          "    void force_linking();\n"
          "}}\n"
          "\n"
          "#endif\n"
      )

      # Add a source file implementing the above function
      set(force_linking_source "${CMAKE_CURRENT_BINARY_DIR}/src/force_linking.cpp")
      FILE(WRITE ${force_linking_source}
          ${copyright}
          "#include <hpx/${name}/force_linking.hpp>\n"
          "\n"
          "namespace hpx { namespace ${name}\n"
          "{\n"
          "    void force_linking() {}\n"
          "}}\n"
          "\n"
      )
  endif()

  foreach(header_file ${headers})
    hpx_debug(${header_file})
  endforeach(header_file)

  add_library(hpx_${name} STATIC
    ${sources} ${force_linking_source}
    ${headers} ${global_header} ${compat_headers}
    ${force_linking_header})

  target_link_libraries(hpx_${name} ${${name}_DEPENDENCIES})
  target_include_directories(hpx_${name} PUBLIC
    $<BUILD_INTERFACE:${HEADER_ROOT}>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>)

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
    ROOT ${SOURCE_ROOT}
    CLASS "Source Files"
    TARGETS ${sources})
  if(HPX_${name_upper}_WITH_COMPATIBILITY_HEADERS)
    add_hpx_source_group(
      NAME hpx_${name}
      ROOT ${COMPAT_HEADER_ROOT}/hpx
      CLASS "Header Files"
      TARGETS ${compat_headers})
  endif()

  if (${name}_GLOBAL_HEADER_GEN)
    add_hpx_source_group(
      NAME hpx_{name}
      ROOT ${CMAKE_CURRENT_BINARY_DIR}/include/hpx
      CLASS "Generated Files"
      TARGETS ${global_header})
  endif()
  if (${name}_FORCE_LINKING_GEN)
    add_hpx_source_group(
      NAME hpx_{name}
      ROOT ${CMAKE_CURRENT_BINARY_DIR}/include/hpx
      CLASS "Generated Files"
      TARGETS ${force_linking_header})
    add_hpx_source_group(
      NAME hpx_{name}
      ROOT ${CMAKE_CURRENT_BINARY_DIR}/src
      CLASS "Generated Files"
      TARGETS ${force_linking_source})
  endif()

  set_target_properties(hpx_${name} PROPERTIES
    FOLDER "Core/Modules"
    POSITION_INDEPENDENT_CODE ON)

  # Install the static library for the module
  if(${name}_INSTALL_BINARIES)
    install(TARGETS hpx_${name} EXPORT HPXTargets
      LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
      ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
      RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
      COMPONENT ${name}
    )
    if(${name}_EXPORT)
      hpx_export_targets(hpx_${name})
    endif()
  endif()

  # Install the headers from the source
  install(
    DIRECTORY include/hpx
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    COMPONENT ${name})

  # Install the compatibility headers from the source
  if(HPX_${name_upper}_WITH_COMPATIBILITY_HEADERS)
    install(
      DIRECTORY include_compatibility/hpx
      DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
      COMPONENT ${name})
  endif()

  write_config_defines_file(
    NAMESPACE ${name_upper}
    FILENAME "${CMAKE_CURRENT_BINARY_DIR}/include/hpx/${name}/config/defines.hpp")

  # Installing the generated header files from the build dir
  install(
    DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/include/hpx
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/hpx
    COMPONENT ${name}
  )

  foreach(dir ${${name}_CMAKE_SUBDIRS})
    add_subdirectory(${dir})
  endforeach(dir)

endfunction(add_hpx_module)
