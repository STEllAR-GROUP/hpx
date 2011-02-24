# Copyright (c) 2007-2009 Hartmut Kaiser
# Copyright (c) 2011 Bryce Lelbach
# Copyright (C) 2007 Douglas Gregor
# Copyright (C) 2007 Troy Straszheim
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying 
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(HPX_UTILS_LOADED TRUE)

if(NOT CMAKE_ALLOW_LOOSE_LOOP_CONSTRUCT)
  set(CMAKE_ALLOW_LOOSE_LOOP_CONSTRUCTS TRUE)
endif()

###############################################################################
# messages 
macro(hpx_info type)
  string(TOLOWER ${type} lctype)
  message("[hpx.info.${lctype}] " ${ARGN})
endmacro()

macro(hpx_debug type)
  if("${HPX_CMAKE_LOGLEVEL}" MATCHES "DEBUG|debug|Debug")
    string(TOLOWER ${type} lctype)
    message("[hpx.debug.${lctype}] " ${ARGN})
  endif()
endmacro()

macro(hpx_warn type)
  if("${HPX_CMAKE_LOGLEVEL}" MATCHES "DEBUG|debug|Debug|WARN|warn|Warn")
    string(TOLOWER ${type} lctype)
    message("[hpx.warn.${lctype}] " ${ARGN})
  endif()
endmacro()

macro(hpx_error type)
  string(TOLOWER ${type} lctype)
  message(FATAL_ERROR "[hpx.error.${lctype}] " ${ARGN})
endmacro()

macro(hpx_message level type)
  if("${level}" MATCHES "ERROR|error|Error")
    string(TOLOWER ${type} lctype)
    hpx_error(${lctype} ${ARGN})
  elseif("${level}" MATCHES "WARN|warn|Warn")
    string(TOLOWER ${type} lctype)
    hpx_warn(${lctype} ${ARGN})
  elseif("${level}" MATCHES "DEBUG|debug|Debug")
    string(TOLOWER ${type} lctype)
    hpx_debug(${lctype} ${ARGN})
  elseif("${level}" MATCHES "INFO|info|Info")
    string(TOLOWER ${type} lctype)
    hpx_info(${lctype} ${ARGN})
  else()
    hpx_error("message" "\"${level}\" is not an HPX configuration logging level.") 
  endif()
endmacro()

macro(hpx_config_loglevel level return)
  set(${return} FALSE)
  if(    "${HPX_CMAKE_LOGLEVEL}" MATCHES "ERROR|error|Error"
     AND "${level}" MATCHES "ERROR|error|Error")
    set(${return} TRUE) 
  elseif("${HPX_CMAKE_LOGLEVEL}" MATCHES "WARN|warn|Warn"
     AND "${level}" MATCHES "WARN|warn|Warn")
    set(${return} TRUE) 
  elseif("${HPX_CMAKE_LOGLEVEL}" MATCHES "DEBUG|debug|Debug"
     AND "${level}" MATCHES "DEBUG|debug|Debug")
    set(${return} TRUE) 
  elseif("${HPX_CMAKE_LOGLEVEL}" MATCHES "INFO|info|Info"
     AND "${level}" MATCHES "INFO|info|Info")
    set(${return} TRUE) 
  endif()
endmacro()

################################################################################
macro(hpx_list_contains var value)
  set(${var})
  foreach(value2 ${ARGN})
    if(${value} STREQUAL ${value2})
      set(${var} TRUE)
    endif()
  endforeach()
endmacro()

###############################################################################
# Print a list 
macro(hpx_print_list level type message list) 
  hpx_config_loglevel(${level} printed)
  if(printed)
    if(${list})
      hpx_message(${level} ${type} "${message}: ")
      foreach(element ${${list}})
        message("    ${element}")
      endforeach()
    else()
      hpx_message(${level} ${type} "${message} is empty.")
    endif()
  endif()
endmacro()

################################################################################
macro(hpx_parse_arguments prefix arg_names option_names)
  set(DEFAULT_ARGS)

  foreach(arg_name ${arg_names})
    set(${prefix}_${arg_name})
  endforeach()

  foreach(option ${option_names})
    set(${prefix}_${option} FALSE)
  endforeach()

  set(current_arg_name DEFAULT_ARGS)
  set(current_arg_list)

  foreach(arg ${ARGN})
    hpx_list_contains(is_arg_name ${arg} ${arg_names})
    if(is_arg_name)
      set(${prefix}_${current_arg_name} ${current_arg_list})
      set(current_arg_name ${arg})
      set(current_arg_list)
    else()
      hpx_list_contains(is_option ${arg} ${option_names})
      if(is_option)
        set(${prefix}_${arg} TRUE)
      else()
        set(current_arg_list ${current_arg_list} ${arg})
      endif()
    endif()
  endforeach()

  set(${prefix}_${current_arg_name} ${current_arg_list})
endmacro()

###############################################################################
# This is an abusive hack that actually installs stuff properly
macro(hpx_install component bin)
  set(install_code
      "file(INSTALL FILES ${CMAKE_CURRENT_BINARY_DIR}/${bin}
            DESTINATION ${CMAKE_INSTALL_PREFIX}/bin
            TYPE EXECUTABLE 
            OPTIONAL
            PERMISSIONS OWNER_READ OWNER_EXECUTE OWNER_WRITE
                        GROUP_READ GROUP_EXECUTE
                        WORLD_READ WORLD_EXECUTE)")
  install(CODE "${install_code}" COMPONENT ${component})
endmacro()

###############################################################################
# Like above, just for components 
macro(hpx_component_install component lib)
  set(install_code
      "file(INSTALL FILES ${CMAKE_CURRENT_BINARY_DIR}/${lib}
            DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
            TYPE SHARED_LIBRARY 
            OPTIONAL
            PERMISSIONS OWNER_READ OWNER_EXECUTE OWNER_WRITE
                        GROUP_READ GROUP_EXECUTE
                        WORLD_READ WORLD_EXECUTE)")
  install(CODE "${install_code}" COMPONENT ${component})
endmacro()

################################################################################
# Installs the ini files for a component, if it was built
macro(hpx_ini_install component name ini)
  set(install_code
      "if(EXISTS "${name}")
          file(INSTALL FILES ${CMAKE_CURRENT_SOURCE_DIR}/${ini}
               DESTINATION ${CMAKE_INSTALL_PREFIX}/share/hpx/ini
               OPTIONAL
               PERMISSIONS OWNER_READ OWNER_WRITE 
                           GROUP_READ 
                           WORLD_READ)
       endif()")
  install(CODE "${install_code}" COMPONENT ${component})
endmacro()

###############################################################################
# This macro builds a HPX component
macro(add_hpx_component name)
  # retrieve arguments
  hpx_parse_arguments(${name}
    "MODULE;SOURCES;HEADERS;DEPENDENCIES;INI" "ESSENTIAL" ${ARGN})

  hpx_print_list("DEBUG" "add_component.${name}" "Sources for ${name}" ${name}_SOURCES)
  hpx_print_list("DEBUG" "add_component.${name}" "Headers for ${name}" ${name}_HEADERS)
  hpx_print_list("DEBUG" "add_component.${name}" "Dependencies for ${name}" ${name}_DEPENDENCIES)
  hpx_print_list("DEBUG" "add_component.${name}" "Configuration files for ${name}" ${name}_INI)

  if(${name}_ESSENTIAL)
    add_library(${name}_component SHARED 
      ${${name}_SOURCES} ${${name}_HEADERS})
  else()
    add_library(${name}_component SHARED EXCLUDE_FROM_ALL
      ${${name}_SOURCES} ${${name}_HEADERS})
  endif() 

  set(prefix "")

  if(NOT MSVC)
    target_link_libraries(${name}_component
      ${${name}_DEPENDENCIES} ${hpx_LIBRARIES} ${BOOST_FOUND_LIBRARIES})
    set(prefix "hpx_component_")
    # main_target is checked by the ini install code, to see if ini files for
    # this component need to be installed
    set(main_target lib${prefix}${name}.so)
    set(install_targets
      lib${prefix}${name}.so
      lib${prefix}${name}.so.${HPX_SOVERSION}
      lib${prefix}${name}.so.${HPX_VERSION})
  else()
    target_link_libraries(${name}_component
      ${${name}_DEPENDENCIES} ${hpx_LIBRARIES} ${BOOST_FOUND_LIBRARIES})
    set(main_target ${name}.dll)
    set(install_targets ${name}.dll)
  endif()

  # set properties of generated shared library
  set_target_properties(${name}_component PROPERTIES
    # create *nix style library versions + symbolic links
    VERSION ${HPX_VERSION}      
    SOVERSION ${HPX_SOVERSION}
    # allow creating static and shared libs without conflicts
    CLEAN_DIRECT_OUTPUT 1 
    OUTPUT_NAME ${prefix}${name})
  
  set_property(TARGET ${name}_component APPEND
               PROPERTY COMPILE_DEFINITIONS
               "HPX_COMPONENT_NAME=${name}"
               "HPX_COMPONENT_EXPORTS")
  
  if(NOT ${name}_MODULE)
    set(${name}_MODULE "Unspecified")
    hpx_warn("add_component.${name}" "Module was not specified for component.")
  endif()

  if(${name}_ESSENTIAL) 
    install(TARGETS ${name}_component
      RUNTIME DESTINATION lib
      ARCHIVE DESTINATION lib
      LIBRARY DESTINATION lib
      COMPONENT ${${name}_MODULE}
      PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE
                  GROUP_READ GROUP_EXECUTE
                  WORLD_READ WORLD_EXECUTE)

    if(${name}_INI)
      install(FILES ${${name}_INI}
        DESTINATION ${CMAKE_INSTALL_PREFIX}/share/hpx/ini
        COMPONENT ${${name}_MODULE}
        PERMISSIONS OWNER_READ OWNER_WRITE 
                    GROUP_READ 
                    WORLD_READ)
    endif()
  else()
    foreach(target ${install_targets})
      hpx_component_install(${${name}_MODULE} ${target})
    endforeach()

    foreach(target ${${name}_INI})
      hpx_ini_install(${${name}_MODULE} ${main_target} ${target})
    endforeach()
  endif()
endmacro()

###############################################################################
# This macro builds a HPX executable
macro(add_hpx_executable name)
  # retrieve arguments
  hpx_parse_arguments(${name}
    "MODULE;SOURCES;HEADERS;DEPENDENCIES" "ESSENTIAL" ${ARGN})

  hpx_print_list("DEBUG" "add_executable.${name}" "Sources for ${name}" ${name}_SOURCES)
  hpx_print_list("DEBUG" "add_executable.${name}" "Headers for ${name}" ${name}_HEADERS)
  hpx_print_list("DEBUG" "add_executable.${name}" "Dependencies for ${name}" ${name}_DEPENDENCIES)

  # add the executable build target
  if(${name}_ESSENTIAL)
    add_executable(${name}_exe 
      ${${name}_SOURCES} ${${name}_HEADERS})
  else()
    add_executable(${name}_exe EXCLUDE_FROM_ALL
      ${${name}_SOURCES} ${${name}_HEADERS})
  endif()

  set_target_properties(${name}_exe PROPERTIES OUTPUT_NAME ${name})

  set_property(TARGET ${name}_exe APPEND
               PROPERTY COMPILE_DEFINITIONS
               "HPX_APPLICATION_NAME=${name}"
               "HPX_APPLICATION_EXPORTS")

  # linker instructions
  target_link_libraries(${name}_exe
    ${${name}_DEPENDENCIES} 
    ${hpx_LIBRARIES}
    ${BOOST_FOUND_LIBRARIES}
    ${pxaccel_LIBRARIES})

  if(NOT ${name}_MODULE)
    set(${name}_MODULE "Unspecified")
    hpx_warn("add_executable.${name}" "Module was not specified for executable.")
  endif()

  if(${name}_ESSENTIAL) 
    install(TARGETS ${name}_exe
      RUNTIME DESTINATION bin
      COMPONENT ${${name}_MODULE}
      PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE
                  GROUP_READ GROUP_EXECUTE
                  WORLD_READ WORLD_EXECUTE)
  else()
    hpx_install(${${name}_MODULE} ${name})
  endif()
endmacro()

###############################################################################
# This macro runs an hpx test 
macro(add_hpx_test target)
  # retrieve arguments
  hpx_parse_arguments(${target}
    "SOURCES;HEADERS;DEPENDENCIES;ARGS" "DONTRUN;DONTCOMPILE" ${ARGN})

  hpx_print_list("DEBUG" "add_test.${target}" "Sources for ${target}" ${target}_SOURCES)
  hpx_print_list("DEBUG" "add_test.${target}" "Headers for ${target}" ${target}_HEADERS)
  hpx_print_list("DEBUG" "add_test.${target}" "Dependencies for ${target}" ${target}_DEPENDENCIES)
  hpx_print_list("DEBUG" "add_test.${target}" "Arguments for ${target}" ${target}_ARGS)

  if(NOT ${target}_DONTCOMPILE)
    add_executable(${target}_test EXCLUDE_FROM_ALL
      ${${target}_SOURCES} ${${target}_HEADERS})
  
    set_property(TARGET ${target}_test APPEND
                 PROPERTY COMPILE_DEFINITIONS
                 "HPX_APPLICATION_NAME=${target}"
                 "HPX_APPLICATION_EXPORTS")
  
    # linker instructions
    target_link_libraries(${target}_test
      ${${target}_DEPENDENCIES} 
      ${hpx_LIBRARIES}
      ${BOOST_FOUND_LIBRARIES}
      ${pxaccel_LIBRARIES})
  
    if(NOT ${target}_DONTRUN)
      add_test(${target}
        ${CMAKE_CURRENT_BINARY_DIR}/${target}_test
        ${${target}_ARGS})
    else()
      hpx_info("add_test.${target}" "Module was not specified for component.")
    endif()
  endif()
endmacro()

###############################################################################
macro(hpx_config_test name var)
  hpx_parse_arguments(${name} "SOURCE;FLAGS" "ESSENTIAL" ${ARGN})

  file(WRITE "${CMAKE_BINARY_DIR}/${CMAKE_FILES_DIRECTORY}/CMakeTmp/src.cpp"
       "${${name}_SOURCE}\n")

  try_compile(${var}
    ${CMAKE_BINARY_DIR}
    ${CMAKE_BINARY_DIR}/${CMAKE_FILES_DIRECTORY}/CMakeTmp/src.cpp
    CMAKE_FLAGS -DCOMPILE_DEFINITIONS:STRING=${${name}_FLAGS}
    OUTPUT_VARIABLE output)

  if(${var})
    set(${var} TRUE CACHE INTERNAL "Test ${name} result.")
    hpx_info("config_test.${name}" "Test passed.")
    file(APPEND ${CMAKE_BINARY_DIR}/${CMAKE_FILES_DIRECTORY}/CMakeOutput.log
      "Test ${name} passed with the following output:\n"
      "${output}\n"
      "Source code was:\n${${name}_SOURCE}\n")
  else()
    set(${var} FALSE CACHE INTERNAL "Test ${name} result.")
    if(${name}_ESSENTIAL)
      hpx_fail("config_test.${name}" "Test failed (check ${CMAKE_BINARY_DIR}/${CMAKE_FILES_DIRECTORY}/CMakeError.log).")
    else()
      hpx_warn("config_test.${name}" "Test failed (check ${CMAKE_BINARY_DIR}/${CMAKE_FILES_DIRECTORY}/CMakeError.log).")
    endif()
    file(APPEND ${CMAKE_BINARY_DIR}/${CMAKE_FILES_DIRECTORY}/CMakeError.log
      "Test ${name} failed with the following output:\n"
      "${output}\n"
      "Source code was:\n${${name}_SOURCE}\n")
  endif()
endmacro()

###############################################################################
macro(hpx_check_pthreads_setaffinity_np variable)
  hpx_config_test(
   "pthreads_setaffinity_np"
   ${variable}
   SOURCE
   "#include <pthread.h>
    
    int f()
    {
        pthread_t th;
        size_t cpusetsize;
        cpu_set_t* cpuset;
        pthread_setaffinity_np(th, cpusetsize, cpuset);
    }
    
    int main()
    {
        return 0;
    }"
    FLAGS -lpthread)
endmacro()

###############################################################################
macro(hpx_force_out_of_tree_build message)
  string(COMPARE EQUAL "${CMAKE_SOURCE_DIR}" "${CMAKE_BINARY_DIR}" insource)
  get_filename_component(parentdir ${CMAKE_SOURCE_DIR} PATH)
  string(COMPARE EQUAL "${CMAKE_SOURCE_DIR}" "${parentdir}" insourcesubdir)
  if(insource OR insourcesubdir)
    hpx_error("in_tree" "${message}")
  endif()
endmacro()


