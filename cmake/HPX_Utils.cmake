# Copyright (c) 2007-2009 Hartmut Kaiser
# Copyright (c) 2011 Bryce Lelbach
# Copyright (C) 2007 Douglas Gregor
# Copyright (C) 2007 Troy Straszheim
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying 
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

if(NOT CMAKE_ALLOW_LOOSE_LOOP_CONSTRUCT)
  set(CMAKE_ALLOW_LOOSE_LOOP_CONSTRUCTS TRUE)
endif()

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
# Print a debug message if HPX_DEBUG is on 
macro(hpx_debug)
  if("${HPX_CMAKE_LOGLEVEL}" STREQUAL "DEBUG")
    message("(HPX.DEBUG) " ${ARGN})
  endif()
endmacro()

###############################################################################
# Print a warning
macro(hpx_warning)
  if("${HPX_CMAKE_LOGLEVEL}" MATCHES "DEBUG|WARN")
    message("(HPX.WARN) " ${ARGN})
  endif()
endmacro()

###############################################################################
# Print a debug message 
macro(hpx_error)
  message(FATAL_ERROR "(HPX.ERROR) " ${ARGN})
endmacro()

###############################################################################
# Convert a list into a delimited string 
macro(hpx_canonicalize_list list string delimiter)
  set(tween "")
  set(collect "${${string}}")

  foreach(element ${list})
    set(realpath ${CMAKE_CURRENT_SOURCE_DIR}/${element})
    set(collect "${collect}${tween}${realpath}")
    if("${tween}" STREQUAL "")
      set(tween "${delimiter}")
    endif()
  endforeach()

  set(${string} "${collect}")
endmacro()

###############################################################################
# Print a list 
macro(hpx_debug_list message list) 
  if("${HPX_CMAKE_LOGLEVEL}" STREQUAL "DEBUG")
    if(${list})
      hpx_debug("${message}: ")
      foreach(element ${${list}})
        message("    ${element}")
      endforeach()
    else()
      hpx_debug("${message} is empty")
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
macro(hpx_install component target_name location)
  get_filename_component(target_path ${target_name} PATH)
  get_filename_component(target_bin ${target_name} NAME)
  get_filename_component(install_path ${location} PATH)
  set(install_code
    "find_path(${target_bin}_was_built
               ${target_bin}
               PATHS ${target_path} NO_DEFAULT_PATH)
     if(${target_bin}_was_built)
       file(INSTALL ${target_name}
            DESTINATION ${install_path}
            PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE
                        GROUP_READ GROUP_EXECUTE
                        WORLD_READ WORLD_EXECUTE)
     else(${target_bin}_was_built)
       message(STATUS \"Not installing: ${location}\")
     endif(${target_bin}_was_built)")
  install(CODE "${install_code}" COMPONENT ${component})
endmacro()

###############################################################################
# Like above, just for components 
macro(hpx_component_install component target_name location)
  if(NOT MSVC)
    get_filename_component(target_path ${target_name} PATH)
    get_filename_component(target_bin ${target_name} NAME)
    get_filename_component(install_path ${location} PATH)
    set(install_code
      "find_path(${target_bin}_was_built
                 ${target_bin}
                 PATHS ${target_path} NO_DEFAULT_PATH)
       if(${target_bin}_was_built)
         file(INSTALL ${target_name}
              DESTINATION ${install_path}
              PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE
                          GROUP_READ GROUP_EXECUTE
                          WORLD_READ WORLD_EXECUTE)
         execute_process(COMMAND \"ln\" \"-fs\"
                         \"${location}.so.${HPX_VERSION}\"
                         \"${location}.so.${HPX_SOVERSION}\")
         execute_process(COMMAND \"ln\" \"-fs\"
                         \"${location}.so.${HPX_VERSION}\"
                         \"${location}.so\")
       else(${target_bin}_was_built)
         message(STATUS \"Not installing: ${location}.so.${HPX_VERSION}\")
         message(STATUS \"Not installing: ${location}.so.${HPX_SOVERSION}\")
         message(STATUS \"Not installing: ${location}.so\")
       endif(${target_bin}_was_built)")
    install(CODE "${install_code}" COMPONENT ${component})
  else()
    get_filename_component(target_path ${target_name} PATH)
    get_filename_component(target_bin ${target_name} NAME)
    get_filename_component(install_path ${location} PATH)
    set(install_code
      "find_path(${target_bin}_was_built
                 ${target_bin}
                 PATHS ${target_path} NO_DEFAULT_PATH)
       if(${target_bin}_was_built)
         file(INSTALL ${target_name}
              DESTINATION ${install_path}
              PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE
                          GROUP_READ GROUP_EXECUTE
                          WORLD_READ WORLD_EXECUTE)
       else(${target_bin}_was_built)
         message(STATUS \"Not installing: ${location}.dll\")
       endif(${target_bin}_was_built)")
    install(CODE "${install_code}" COMPONENT ${component})
  endif()
endmacro()

################################################################################
# Installs the ini files for a component, if it was built
macro(hpx_ini_install component target_name location ini_files)
  get_filename_component(target_path ${target_name} PATH)
  get_filename_component(target_bin ${target_name} NAME)
  hpx_canonicalize_list(${ini_files} serialized_ini_files " ")
  set(install_code
    "find_path(${target_bin}_was_built
               ${target_bin}
               PATHS ${target_path} NO_DEFAULT_PATH)
     if(${target_bin}_was_built)
       file(INSTALL ${serialized_ini_files}
            DESTINATION ${CMAKE_INSTALL_PREFIX}/share/hpx/ini
            PERMISSIONS OWNER_READ OWNER_WRITE 
                        GROUP_READ 
                        WORLD_READ)
     else(${target_bin}_was_built)
       message(STATUS \"Not installing: ${location}.so.${HPX_VERSION}'s configuration files\")
     endif(${target_bin}_was_built)")
  install(CODE "${install_code}" COMPONENT ${component})
endmacro()

###############################################################################
# This macro builds a HPX component
macro(add_hpx_component name)
  # retrieve arguments
  hpx_parse_arguments(${name}
    "MODULE;SOURCES;HEADERS;DEPENDENCIES;INI" "ESSENTIAL" ${ARGN})

  hpx_debug_list("Sources for ${name}" ${name}_SOURCES)
  hpx_debug_list("Headers for ${name}" ${name}_HEADERS)
  hpx_debug_list("Dependencies for ${name}" ${name}_DEPENDENCIES)
  hpx_debug_list("Configuration files for ${name}" ${name}_INI)

  # add defines for this component
  add_definitions(-DHPX_COMPONENT_NAME=${name})
  add_definitions(-DHPX_COMPONENT_EXPORTS)

  if(${name}_ESSENTIAL)
    add_library(${name}_component SHARED 
      ${${name}_SOURCES} ${${name}_HEADERS})
  else()
    add_library(${name}_component SHARED EXCLUDE_FROM_ALL
      ${${name}_SOURCES} ${${name}_HEADERS})
  endif() 

  # set properties of generated shared library
  set_target_properties(${name}_component PROPERTIES
    # create *nix style library versions + symbolic links
    VERSION ${HPX_VERSION}      
    SOVERSION ${HPX_SOVERSION}
    # allow creating static and shared libs without conflicts
    CLEAN_DIRECT_OUTPUT 1 
    OUTPUT_NAME ${hpx_COMPONENT_LIBRARY_PREFIX}${name})

  target_link_libraries(${name}_component
    ${${name}_DEPENDENCIES} ${hpx_LIBRARIES} ${BOOST_FOUND_LIBRARIES})

  if(NOT MSVC)
    set(installed_target
      ${CMAKE_INSTALL_PREFIX}/lib/lib${hpx_COMPONENT_LIBRARY_PREFIX}${name})
    set(built_target
      ${CMAKE_CURRENT_BINARY_DIR}/lib${hpx_COMPONENT_LIBRARY_PREFIX}${name}.so.${HPX_VERSION})
  else()
    set(installed_target
      ${CMAKE_INSTALL_PREFIX}/lib/${hpx_COMPONENT_LIBRARY_PREFIX}${name})
    set(built_target
      ${CMAKE_CURRENT_BINARY_DIR}/${hpx_COMPONENT_LIBRARY_PREFIX}${name}.dll)
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
    hpx_component_install(${${name}_MODULE}
      ${built_target} ${installed_target})
  
    if(${name}_INI)
      hpx_ini_install(${${name}_MODULE}
        ${built_target} ${installed_target} ${${name}_INI})
    endif()
  endif()
endmacro()

###############################################################################
# This macro builds a HPX executable
macro(add_hpx_executable name)
  # retrieve arguments
  hpx_parse_arguments(${name}
    "MODULE;SOURCES;HEADERS;DEPENDENCIES" "ESSENTIAL" ${ARGN})

  hpx_debug_list("Sources for ${name}" ${name}_SOURCES)
  hpx_debug_list("Headers for ${name}" ${name}_HEADERS)
  hpx_debug_list("Dependencies for ${name}" ${name}_DEPENDENCIES)

  # add defines for this executable
  add_definitions(-DHPX_APPLICATION_EXPORTS)

  # add the executable build target
  if(${name}_ESSENTIAL)
    add_executable(${name}_exe 
      ${${name}_SOURCES} ${${name}_HEADERS})
  else()
    add_executable(${name}_exe EXCLUDE_FROM_ALL
      ${${name}_SOURCES} ${${name}_HEADERS})
  endif()

  # avoid conflicts between source and binary target names - this is done
  # automatically for shared libraries if CMAKE_DEBUG_POSTFIX is defined, but
  # not for executables.
  set_target_properties(${name}_exe PROPERTIES
    DEBUG_OUTPUT_NAME ${name}${CMAKE_DEBUG_POSTFIX}
    RELEASE_OUTPUT_NAME ${name}
    RELWITHDEBINFO_OUTPUT_NAME ${name}
    MINSIZEREL_OUTPUT_NAME ${name})

  # linker instructions
  target_link_libraries(${name}_exe
    ${${name}_DEPENDENCIES} 
    ${hpx_LIBRARIES}
    ${BOOST_FOUND_LIBRARIES}
    ${pxaccel_LIBRARIES})

  if(${name}_ESSENTIAL) 
    install(TARGETS ${name}_exe
      RUNTIME DESTINATION bin
      COMPONENT ${${name}_MODULE}
      PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE
                  GROUP_READ GROUP_EXECUTE
                  WORLD_READ WORLD_EXECUTE)
  else()
    if("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
      hpx_install(${${name}_MODULE} ${CMAKE_CURRENT_BINARY_DIR}/${name}
        ${CMAKE_INSTALL_PREFIX}/bin/${name}${CMAKE_DEBUG_POSTFIX}) 
    else()
      hpx_install(${${name}_MODULE} ${CMAKE_CURRENT_BINARY_DIR}/${name}
        ${CMAKE_INSTALL_PREFIX}/bin/${name}) 
    endif()
  endif()
endmacro()

###############################################################################
# This macro runs an hpx test 
macro(add_hpx_test target name)
  # retrieve arguments
  hpx_parse_arguments(${target}
    "SOURCES;HEADERS;DEPENDENCIES;ARGS" "DONTRUN;DONTCOMPILE" ${ARGN})

  hpx_debug_list("Sources for ${target}" ${target}_SOURCES)
  hpx_debug_list("Headers for ${target}" ${target}_HEADERS)
  hpx_debug_list("Dependencies for ${target}" ${target}_DEPENDENCIES)
  hpx_debug_list("Arguments for ${target}" ${target}_ARGS)

  if(NOT ${target}_DONTCOMPILE)
    # add defines for this executable
    add_definitions(-DHPX_APPLICATION_EXPORTS)

    add_executable(${target}_test EXCLUDE_FROM_ALL
      ${${target}_SOURCES} ${${target}_HEADERS})
  
    # linker instructions
    target_link_libraries(${target}_test
      ${${target}_DEPENDENCIES} 
      ${hpx_LIBRARIES}
      ${BOOST_FOUND_LIBRARIES}
      ${pxaccel_LIBRARIES})
  
    if(NOT ${target}_DONTRUN)
      add_test(${name}
        ${CMAKE_CURRENT_BINARY_DIR}/${target}_test
        ${${target}_ARGS})
    endif()
  endif()
endmacro()

###############################################################################
macro(hpx_check_pthreads_setaffinity_np variable)
  set(CMAKE_REQUIRED_FLAGS -lpthread)

  include(CheckCSourceCompiles)

  check_c_source_compiles(
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
    }" ${variable})
endmacro()

###############################################################################
macro(hpx_force_out_of_tree_build message)
  string(COMPARE EQUAL "${CMAKE_SOURCE_DIR}" "${CMAKE_BINARY_DIR}" insource)
  get_filename_component(parentdir ${CMAKE_SOURCE_DIR} PATH)
  string(COMPARE EQUAL "${CMAKE_SOURCE_DIR}" "${parentdir}" insourcesubdir)
  if(insource OR insourcesubdir)
    message(FATAL_ERROR "${message}")
  endif()
endmacro()

