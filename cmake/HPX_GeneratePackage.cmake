# Copyright (c) 2018 Christopher Hinz
# Copyright (c) 2014 Thomas Heller
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

include(CMakePackageConfigHelpers)

set(CMAKE_DIR "cmake-${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION}" CACHE STRING "directory (in share), where to put FindHPX cmake module")

write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/lib/cmake/${HPX_PACKAGE_NAME}/${HPX_PACKAGE_NAME}ConfigVersion.cmake"
  VERSION ${HPX_VERSION}
  COMPATIBILITY AnyNewerVersion
)

export(TARGETS ${HPX_EXPORT_TARGETS}
  FILE "${CMAKE_CURRENT_BINARY_DIR}/lib/cmake/${HPX_PACKAGE_NAME}/HPXTargets.cmake"
#  NAMESPACE hpx::
)

# https://github.com/boost-cmake/bcm/blob/master/share/bcm/cmake/BCMPkgConfig.cmake
# https://gitlab.kitware.com/cmake/cmake/issues/17984

function(hpx_collect_usage_requirements target compile_definitions compile_options requires_pic include_directories system_include_directories link_libraries link_options already_processed_targets)
  if(NOT TARGET ${target})
    message(ERROR "'${target}' should be a target.")
  endif()

  set(_already_processed_targets ${${already_processed_targets}} ${target})

  get_property(target_compile_definitions TARGET ${target} PROPERTY INTERFACE_COMPILE_DEFINITIONS)

  get_property(target_compile_options TARGET ${target} PROPERTY INTERFACE_COMPILE_OPTIONS)

  get_property(target_requires_pic TARGET ${target} PROPERTY INTERFACE_POSITION_INDEPENDENT_CODE)

  get_property(target_include_directories TARGET ${target} PROPERTY INTERFACE_INCLUDE_DIRECTORIES)

  get_property(target_system_include_directories TARGET ${target} PROPERTY INTERFACE_SYSTEM_INCLUDE_DIRECTORIES)

  get_property(target_link_libraries TARGET ${target} PROPERTY INTERFACE_LINK_LIBRARIES)

  get_property(target_link_options TARGET ${target} PROPERTY INTERFACE_LINK_OPTIONS)

  get_property(target_type TARGET ${target} PROPERTY TYPE)

  if(${target_type} STREQUAL "STATIC_LIBRARY" OR ${target_type} STREQUAL "SHARED_LIBRARY")
    set(libraries ${libraries} $<TARGET_FILE:${target}>)
  endif()

  foreach(dep IN LISTS target_link_libraries)
    if(dep MATCHES "^\\$<LINK_ONLY:([^>]+)>$")
      # This a private link dependency. Do not inherit the target's usage requirements.

      set(dep_target ${CMAKE_MATCH_1})

      if (TARGET ${dep_target})
        message(ERROR "'${dep_target}' should be a target.")
      endif()

      get_property(dep_type TARGET ${dep_target} PROPERTY TYPE)

      # If the target is a library link against it.
      if(${dep_type} STREQUAL "STATIC_LIBRARY" OR ${dep_type} STREQUAL "SHARED_LIBRARY")
        set(libraries ${libraries} $<TARGET_FILE:${dep_target}>)
      endif()
    elseif(TARGET ${dep})
      # This is a public dependency. Follow the dependency graph and add the target's usage requirements.

      if (NOT dep IN_LIST _already_processed_targets)
        set(dep_compile_definitions)
        set(dep_compile_options)
        set(dep_requires_pic)
        set(dep_include_directories)
        set(dep_system_include_directories)
        set(dep_link_libraries)
        set(dep_link_options)

        hpx_collect_usage_requirements(${dep} dep_compile_definitions
                                              dep_compile_options
                                              dep_requires_pic
                                              dep_include_directories
                                              dep_system_include_directories
                                              dep_link_libraries
                                              dep_link_options
                                              _already_processed_targets)

        set(target_compile_definitions ${target_compile_definitions} ${dep_compile_definitions})
        set(target_compile_options ${target_compile_options} ${dep_compile_options})
        set(target_include_directories ${target_include_directories} ${dep_include_directories})
        set(target_system_include_directories ${target_system_include_directories} ${dep_system_include_directories})
        set(libraries ${libraries} ${dep_link_libraries})
        set(target_link_options ${target_link_options} ${dep_link_options})

        if(dep_requires_pic)
          set(target_requires_pic ON)
        endif()
      endif()
    elseif(dep MATCHES "\\$<")
      # This is a plain generator expression. As we can not determine its type at this point,
      # just assume that it is an absolute path.
      set(libraries ${libraries} ${dep})
    elseif(dep MATCHES "^-l")
      # This is a library with a linker flag.
      set(libraries ${libraries} ${dep})
    else()
      # This is a plain path.
      if (IS_ABSOLUTE ${dep})
        set(libraries ${libraries} ${dep})
      else()
        set(libraries ${libraries} -l${dep})
      endif()
    endif()
  endforeach()

  set(${compile_definitions} ${target_compile_definitions} PARENT_SCOPE)
  set(${compile_options} ${target_compile_options} PARENT_SCOPE)
  set(${requires_pic} ${target_requires_pic} PARENT_SCOPE)
  set(${include_directories} ${target_include_directories} PARENT_SCOPE)
  set(${system_include_directories} ${target_system_include_directories} PARENT_SCOPE)
  set(${link_libraries} ${libraries} PARENT_SCOPE)
  set(${link_options} ${target_link_options} PARENT_SCOPE)

  set(${already_processed_targets} ${_already_processed_targets} PARENT_SCOPE)
endfunction()

function(hpx_sanitize_usage_requirements property sanitized_property is_build)
  foreach(value IN LISTS ${property})
    if(is_build)
      string(REPLACE "$<BUILD_INTERFACE:" "$<1:" value "${value}")
      string(REPLACE "$<INSTALL_INTERFACE:" "$<0:" value "${value}")
    else()
      string(REPLACE "$<BUILD_INTERFACE:" "$<0:" value "${value}")
      string(REPLACE "$<INSTALL_INTERFACE:" "$<1:${CMAKE_INSTALL_PREFIX}/" value "${value}")
    endif()

    set(_sanitized_property ${_sanitized_property} ${value})
  endforeach()

  set(${sanitized_property} ${_sanitized_property} PARENT_SCOPE)
endfunction()

function(hpx_construct_cflag_list compile_definitions compile_options requires_pic include_directories system_include_directories cflag_list)
  set(_cflag_list "${_cflag_list} $<$<BOOL:${${compile_definitions}}>:-D$<JOIN:${${compile_definitions}}, -D>>")

  set(_cflag_list "${_cflag_list} $<JOIN:${${compile_options}}, >")

  if (requires_pic)
    set(_cflag_list "${_cflag_list} -fPIC")
  endif()

  set(_cflag_list "${_cflag_list} $<$<BOOL:${${include_directories}}>:-I$<JOIN:${${include_directories}}, -I>>")

  set(_cflag_list "${_cflag_list} $<$<BOOL:${${system_include_directories}}>:-isystem$<JOIN:${${system_include_directories}}, -isystem>>")

  set(${cflag_list} ${_cflag_list} PARENT_SCOPE)
endfunction()

function(hpx_construct_library_list link_libraries link_options library_list)
  foreach(library IN LISTS ${link_libraries})
    set(_library_list "${_library_list} ${library}")
  endforeach()

  foreach(option IN LISTS ${link_options})
    set(_library_list "${_library_list} ${option}")
  endforeach()

  set(${library_list} ${_library_list} PARENT_SCOPE)
endfunction()

function(hpx_generate_pkgconfig_from_target target template is_build)
  if(is_build)
    set(HPX_CONF_PREFIX ${CMAKE_BINARY_DIR})
    set(OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/lib/pkgconfig/)
  else()
    set(HPX_CONF_PREFIX ${CMAKE_INSTALL_PREFIX})
    set(OUTPUT_DIR ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY})
  endif()

  hpx_collect_usage_requirements(${target} hpx_compile_definitions
                                           hpx_compile_options
                                           hpx_requires_pic
                                           hpx_include_directories
                                           hpx_system_include_directories
                                           hpx_link_libraries
                                           hpx_link_options
                                           processed_targets)

  hpx_sanitize_usage_requirements(hpx_compile_definitions hpx_compile_definitions is_build)
  hpx_sanitize_usage_requirements(hpx_compile_options hpx_compile_options is_build)
  hpx_sanitize_usage_requirements(hpx_include_directories hpx_include_directories is_build)
  hpx_sanitize_usage_requirements(hpx_system_include_directories hpx_system_include_directories is_build)
  hpx_sanitize_usage_requirements(hpx_link_libraries hpx_link_libraries is_build)
  hpx_sanitize_usage_requirements(hpx_link_options hpx_link_options is_build)

  hpx_construct_cflag_list(hpx_compile_definitions hpx_compile_options hpx_requires_pic hpx_include_directories hpx_system_include_directories hpx_cflags_list)
  hpx_construct_library_list(hpx_link_libraries hpx_link_options hpx_library_list)

  configure_file(cmake/templates/${template}.pc.in ${CMAKE_BINARY_DIR}/${template}.pc.in @ONLY ESCAPE_QUOTES)

  file(GENERATE OUTPUT ${OUTPUT_DIR}/${template}$<$<CONFIG:Debug>:_debug>.pc
                INPUT ${CMAKE_BINARY_DIR}/${template}.pc.in)
endfunction()

hpx_generate_pkgconfig_from_target(hpx::application hpx_application FALSE)
hpx_generate_pkgconfig_from_target(hpx::application hpx_application TRUE)

hpx_generate_pkgconfig_from_target(hpx::component hpx_component FALSE)
hpx_generate_pkgconfig_from_target(hpx::component hpx_component TRUE)

## Generate library list for pkg config ...
#set(_is_debug FALSE)
#set(_is_release FALSE)
#foreach(lib ${HPX_LIBRARIES})
#  if(lib STREQUAL "debug")
#    set(_is_debug TRUE)
#    set(_is_release FALSE)
#  elseif(lib STREQUAL "optimized")
#    set(_is_debug FALSE)
#    set(_is_release TRUE)
#  elseif(lib STREQUAL "general")
#    set(_is_debug FALSE)
#    set(_is_release FALSE)
#  else()
#    if(NOT EXISTS "${lib}")
#      set(lib "-l${lib}")
#    endif()
#    if(_is_debug)
#      set(HPX_PKG_DEBUG_LIBRARIES "${HPX_PKG_DEBUG_LIBRARIES} ${lib}")
#    elseif(_is_release)
#      set(HPX_PKG_LIBRARIES "${HPX_PKG_LIBRARIES} ${lib}")
#    else()
#      set(HPX_PKG_LIBRARIES "${HPX_PKG_LIBRARIES} ${lib}")
#      set(HPX_PKG_DEBUG_LIBRARIES "${HPX_PKG_DEBUG_LIBRARIES} ${lib}")
#    endif()
#    set(_is_debug FALSE)
#    set(_is_release FALSE)
#  endif()
#endforeach()
#
#set(HPX_PKG_LIBRARY_DIR "")
#foreach(dir ${HPX_LIBRARY_DIR})
#  set(HPX_PKG_LIBRARY_DIR "${HPX_PKG_LIBRARY_DIR} -L${dir}")
#endforeach()
#
#if(HPX_WITH_STATIC_LINKING)
#  set(HPX_PKG_LIBRARIES_BAZEL "${HPX_CONF_PREFIX}/lib/libhpx.a ${HPX_PKG_LIBRARIES}")
#  set(HPX_PKG_DEBUG_LIBRARIES_BAZEL "${HPX_CONF_PREFIX}/lib/libhpxd.a ${HPX_PKG_DEBUG_LIBRARIES}")
#  set(HPX_CONF_LIBRARIES "general;hpx;${HPX_LIBRARIES}")
#  set(HPX_PKG_LIBRARIES "\${libdir}/libhpx.a ${HPX_PKG_LIBRARIES}")
#  set(HPX_PKG_DEBUG_LIBRARIES "\${libdir}/libhpxd.a ${HPX_PKG_DEBUG_LIBRARIES}")
#else()
#  set(HPX_CONF_LIBRARIES "general;hpx_init;general;hpx;${HPX_LIBRARIES}")
#  if(APPLE)
#    set(HPX_PKG_LIBRARIES_BAZEL "${HPX_CONF_PREFIX}/lib/libhpx_init.a ${HPX_CONF_PREFIX}/lib/libhpx.dylib ${HPX_PKG_LIBRARIES}")
#    set(HPX_PKG_DEBUG_LIBRARIES_BAZEL "${HPX_CONF_PREFIX}/lib/libhpx_initd.a ${HPX_CONF_PREFIX}/lib/libhpxd.dylib ${HPX_PKG_DEBUG_LIBRARIES}")
#    set(HPX_PKG_LIBRARIES "\${libdir}/libhpx_init.a \${libdir}/libhpx.dylib ${HPX_PKG_LIBRARIES}")
#    set(HPX_PKG_DEBUG_LIBRARIES "\${libdir}/libhpx_initd.a \${libdir}/libhpxd.dylib ${HPX_PKG_DEBUG_LIBRARIES}")
#  else()
#    set(HPX_PKG_LIBRARIES_BAZEL "${HPX_CONF_PREFIX}/lib/libhpx_init.a ${HPX_CONF_PREFIX}/lib/libhpx.so ${HPX_PKG_LIBRARIES}")
#    set(HPX_PKG_DEBUG_LIBRARIES_BAZEL "${HPX_CONF_PREFIX}/lib/libhpx_initd.a ${HPX_CONF_PREFIX}/lib/libhpxd.so ${HPX_PKG_DEBUG_LIBRARIES}")
#    set(HPX_PKG_LIBRARIES "\${libdir}/libhpx_init.a \${libdir}/libhpx.so ${HPX_PKG_LIBRARIES}")
#    set(HPX_PKG_DEBUG_LIBRARIES "\${libdir}/libhpx_initd.a \${libdir}/libhpxd.so ${HPX_PKG_DEBUG_LIBRARIES}")
#  endif()
#endif()
#
#if(HPX_WITH_DYNAMIC_HPX_MAIN AND (("${CMAKE_SYSTEM_NAME}" STREQUAL "Linux") OR (APPLE)))
#  set(HPX_LINK_LIBRARIES "general;hpx_wrap;")
#endif()
#
#if(HPX_WITH_DYNAMIC_HPX_MAIN AND (("${CMAKE_SYSTEM_NAME}" STREQUAL "Linux") OR (APPLE)))
#    set(HPX_PKG_DEBUG_LINK_LIBRARIES "\${libdir}/libhpx_wrapd.a")
#    set(HPX_PKG_LINK_LIBRARIES "\${libdir}/libhpx_wrap.a")
#endif()
#
#set(HPX_LINKER_FLAGS "")
#if(HPX_WITH_DYNAMIC_HPX_MAIN)
#    if("${CMAKE_SYSTEM_NAME}" STREQUAL "Linux")
#        set(HPX_LINKER_FLAGS "-Wl,-wrap=main")
#    elseif(APPLE)
#        set(HPX_LINKER_FLAGS "-Wl,-e,_initialize_main")
#    endif()
#endif()
#
## Get the include directories we need ...
#get_directory_property(_INCLUDE_DIRS INCLUDE_DIRECTORIES)
#
## replace all characters with special regex meaning
#set(special_chars "^;+;*;?;$;.;-;|;(;);]")
#set(binarydir_escaped ${CMAKE_BINARY_DIR})
#set(sourcedir_escaped ${PROJECT_SOURCE_DIR})
#foreach(special_char ${special_chars})
#  string(REPLACE "${special_char}" "\\${special_char}" binarydir_escaped ${binarydir_escaped})
#  string(REPLACE "${special_char}" "\\${special_char}" sourcedir_escaped ${sourcedir_escaped})
#endforeach()
#
## '[' has special meaning in lists
#string(REPLACE "[" "\\[" binarydir_escaped ${binarydir_escaped})
#string(REPLACE "[" "\\[" sourcedir_escaped ${sourcedir_escaped})
#
#foreach(dir ${_INCLUDE_DIRS})
#  if((NOT dir MATCHES "^${binarydir_escaped}.*")
#    AND (NOT dir MATCHES "^${sourcedir_escaped}.*"))
#    set(_NEEDED_INCLUDE_DIRS "${_NEEDED_INCLUDE_DIRS} -I${dir}")
#    set(_NEEDED_CMAKE_INCLUDE_DIRS ${_NEEDED_CMAKE_INCLUDE_DIRS} "${dir}")
#  else()
#    set(_NEEDED_BUILD_DIR_INCLUDE_DIRS "${_NEEDED_BUILD_DIR_INCLUDE_DIRS} -I${dir}")
#    set(_NEEDED_CMAKE_BUILD_DIR_INCLUDE_DIRS ${_NEEDED_CMAKE_BUILD_DIR_INCLUDE_DIRS} "${dir}")
#  endif()
#endforeach()
#
#set(hpx_bazel_file cmake/templates/hpx_bazel_defs.bzl.in)
#if(_is_debug)
#  set(hpx_bazel_file cmake/templates/hpx_bazel_defs_debug.bzl.in)
#endif()
#
## Configure config for the install dir ...
#set(HPX_CONF_INCLUDE_DIRS
#  "-I${CMAKE_INSTALL_PREFIX}/include -I${CMAKE_INSTALL_PREFIX}/include/hpx/external ${_NEEDED_INCLUDE_DIRS}"
#)
#set(HPX_CMAKE_CONF_INCLUDE_DIRS
#  "${CMAKE_INSTALL_PREFIX}/include"
#  "${CMAKE_INSTALL_PREFIX}/include/hpx/external"
#  "${_NEEDED_CMAKE_INCLUDE_DIRS}"
#)
#set(HPX_CONF_PREFIX ${CMAKE_INSTALL_PREFIX})
#set(HPX_CONF_LIBRARY_DIR ${HPX_LIBRARY_DIR})
#
configure_file(cmake/templates/${HPX_PACKAGE_NAME}Config.cmake.in
  "${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/${HPX_PACKAGE_NAME}Config.cmake"
  ESCAPE_QUOTES @ONLY)
#configure_file(cmake/templates/hpx_application.pc.in
#  "${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/hpx_application.pc"
#  ESCAPE_QUOTES @ONLY)
#configure_file(cmake/templates/hpx_application_debug.pc.in
#  "${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/hpx_application_debug.pc"
#  ESCAPE_QUOTES @ONLY)
#configure_file(cmake/templates/hpx_component.pc.in
#  "${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/hpx_component.pc"
#  ESCAPE_QUOTES @ONLY)
#configure_file(cmake/templates/hpx_component_debug.pc.in
#  "${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/hpx_component_debug.pc"
#  ESCAPE_QUOTES @ONLY)
## Configure hpxcxx
configure_file(cmake/templates/hpxcxx.in
  "${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/hpxcxx"
  @ONLY)
#
## Configure bazel file
#configure_file(${hpx_bazel_file}
#  "${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/hpx_bazel_defs.bzl"
#  ESCAPE_QUOTES @ONLY)
#
## ... and the build dir
#set(HPX_CONF_PREFIX "${CMAKE_BINARY_DIR}")
#set(HPX_CONF_LIBDIR "lib")
#set(HPX_CONF_INCLUDEDIR "include")
#set(HPX_CONF_INCLUDE_DIRS
#  "${_NEEDED_BUILD_DIR_INCLUDE_DIRS} ${_NEEDED_INCLUDE_DIRS}"
#)
#set(HPX_CMAKE_CONF_INCLUDE_DIRS
#  ${_NEEDED_CMAKE_BUILD_DIR_INCLUDE_DIRS}
#  ${_NEEDED_CMAKE_INCLUDE_DIRS}
#)
configure_file(cmake/templates/${HPX_PACKAGE_NAME}Config.cmake.in
  "${CMAKE_CURRENT_BINARY_DIR}/lib/cmake/${HPX_PACKAGE_NAME}/${HPX_PACKAGE_NAME}Config.cmake"
  ESCAPE_QUOTES @ONLY)
#configure_file(cmake/templates/hpx_application.pc.in
#  "${CMAKE_CURRENT_BINARY_DIR}/lib/pkgconfig/hpx_application.pc"
#  ESCAPE_QUOTES @ONLY)
#configure_file(cmake/templates/hpx_application_debug.pc.in
#  "${CMAKE_CURRENT_BINARY_DIR}/lib/pkgconfig/hpx_application_debug.pc"
#  ESCAPE_QUOTES @ONLY)
#configure_file(cmake/templates/hpx_component.pc.in
#  "${CMAKE_CURRENT_BINARY_DIR}/lib/pkgconfig/hpx_component.pc"
#  ESCAPE_QUOTES @ONLY)
#configure_file(cmake/templates/hpx_component_debug.pc.in
#  "${CMAKE_CURRENT_BINARY_DIR}/lib/pkgconfig/hpx_component_debug.pc"
#  ESCAPE_QUOTES @ONLY)
## Configure hpxcxx
configure_file(cmake/templates/hpxcxx.in
  "${CMAKE_CURRENT_BINARY_DIR}/bin/hpxcxx"
  @ONLY)
#
## Configure bazel file
#configure_file(${hpx_bazel_file}
#   "${CMAKE_CURRENT_BINARY_DIR}/lib/bazel/hpx_bazel_defs.bzl"
#  @ONLY)

# Configure macros for the install dir ...
set(HPX_CMAKE_MODULE_PATH "${CMAKE_INSTALL_FULL_LIBDIR}/cmake/${HPX_PACKAGE_NAME}")
configure_file(cmake/templates/HPXMacros.cmake.in
  "${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/HPXMacros.cmake"
  ESCAPE_QUOTES @ONLY)
# ... and the build dir
set(HPX_CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")
configure_file(cmake/templates/HPXMacros.cmake.in
  "${CMAKE_CURRENT_BINARY_DIR}/lib/cmake/${HPX_PACKAGE_NAME}/HPXMacros.cmake"
  ESCAPE_QUOTES @ONLY)

install(
  EXPORT HPXTargets
  FILE HPXTargets.cmake
#  NAMESPACE hpx::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${HPX_PACKAGE_NAME}
)

install(
  FILES
    "${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/${HPX_PACKAGE_NAME}Config.cmake"
    "${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/HPXMacros.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/lib/cmake/${HPX_PACKAGE_NAME}/${HPX_PACKAGE_NAME}ConfigVersion.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${HPX_PACKAGE_NAME}
  COMPONENT cmake
)

install(
  FILES
    "${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/hpx_application.pc"
    "${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/hpx_application_debug.pc"
    "${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/hpx_component.pc"
    "${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/hpx_component_debug.pc"
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/pkgconfig
  COMPONENT pkgconfig
)

install(
  FILES
    "${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/hpxcxx"
  DESTINATION ${CMAKE_INSTALL_BINDIR}
  COMPONENT compiler_wrapper
  PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE
  GROUP_READ GROUP_EXECUTE
  WORLD_READ WORLD_EXECUTE
)

install(
  FILES
    "${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/hpx_bazel_defs.bzl"
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/bazel
  COMPONENT bazel
)
