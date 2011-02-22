# Copyright (c) 2007-2011 Hartmut Kaiser
# Copyright (c) 2010-2011 Matt Anderson
# Copyright (c) 2011      Bryce Lelbach
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying 
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

if(NOT CMAKE_ALLOW_LOOSE_LOOP_CONSTRUCT)
  set(CMAKE_ALLOW_LOOSE_LOOP_CONSTRUCTS TRUE)
endif()

set(HPX_FINDPACKAGE_LOADED TRUE)

if(NOT HPX_UTILS_LOADED)
  include(HPX_Utils)
endif()

macro(hpx_get_version name)
  if(NOT ${name}_VERSION_SEARCHED)
  set(${name}_VERSION_SEARCHED ON CACHE INTERNAL "Searched for ${name} library version.")

  hpx_parse_arguments(${name}
    "HEADERS;HEADER_PATHS;LIBRARIES;LIBRARY_PATHS" "ESSENTIAL" ${ARGN})
  
  #############################################################################
  # Check if ${name}_ROOT is defined and use that path first if
  # ${name}_USE_SYSTEM is defined.
  if(NOT ${name}_USE_SYSTEM)
    if(NOT ${name}_ROOT AND NOT $ENV{${name}_ROOT} STREQUAL "")
      set(${name}_ROOT $ENV{${name}_ROOT})
    endif(NOT ${name}_ROOT AND NOT $ENV{${name}_ROOT} STREQUAL "")
  endif()
  
  set(rooted_header_paths)
  foreach(path ${${name}_HEADER_PATHS})
    list(APPEND rooted_header_paths ${${name}_ROOT}/${path})
  endforeach()

  if(${name}_ROOT)
    find_path(${name}_INCLUDE_DIR
      NAMES ${${name}_HEADERS}
      PATHS ${rooted_header_paths}
      NO_DEFAULT_PATH)

    if(NOT ${name}_INCLUDE_DIR)
      hpx_warn("get_version.${name}" "Header not found in ${${name}_ROOT}, trying system path.")
      unset(${name}_ROOT)
    else()
      hpx_info("get_version.${name}" "Header found in ${${name}_ROOT}.") 
    endif()
  endif()

  # if not found, retry using system path
  if(NOT ${name}_ROOT)
    find_path(${name}_INCLUDE_DIR NAMES ${${name}_HEADERS})
    
    if(NOT ${name}_INCLUDE_DIR)
      if(${name}_ESSENTIAL)
        hpx_error("get_version.${name}" "Header not found in system path.")
      else() 
        hpx_warn("get_version.${name}" "Header not found in system path.")
      endif()
      unset(${name}_ROOT)
    else()
      hpx_info("get_version.${name}" "Header found in system path.") 
    endif()
  endif()
  
#  file(READ "${BOOST_VERSION_HPP}" BOOST_VERSION_HPP_CONTENTS)
  
#  string(REGEX REPLACE ".*#define BOOST_LIB_VERSION \"([0-9_]+)\".*" "\\1"
#    BOOST_VERSION "${BOOST_VERSION_HPP_CONTENTS}")
#  string(REGEX REPLACE ".*#define BOOST_VERSION ([0-9]+).*" "\\1"
#    BOOST_VERSION_NUM "${BOOST_VERSION_HPP_CONTENTS}")
    
#  if(NOT "${BOOST_VERSION_NUM}" STREQUAL "0")
#    math(EXPR BOOST_MAJOR_VERSION "${BOOST_VERSION_NUM} / 100000")
#    math(EXPR BOOST_MINOR_VERSION "${BOOST_VERSION_NUM} / 100 % 1000")
#    math(EXPR BOOST_PATCH_VERSION "${BOOST_VERSION_NUM} % 100")
#  else()
#    hpx_error("boost.version" "Invalid Boost version ${BOOST_VERSION_NUM}.")
#  endif()

  endif()
endmacro()

macro(hpx_find_package name)
  if(NOT ${name}_SEARCHED)
  set(${name}_SEARCHED ON CACHE INTERNAL "Searched for ${name} library.")

  if(NOT ${name}_VERSION_SEARCHED)
    hpx_get_version(${name} ${ARGN})
  endif()

  set(rooted_lib_paths)
  foreach(path ${${name}_LIBRARY_PATHS})
    list(APPEND rooted_lib_paths ${${name}_ROOT}/${path})
  endforeach()

  if(${name}_ROOT)
    find_library(${name}_LIBRARY
      NAMES ${${name}_LIBRARIES}
      PATHS ${rooted_lib_paths}
      NO_DEFAULT_PATH) 

    if(NOT ${name}_LIBRARY)
      hpx_warn("find_library.${name}" "Library not found in ${${name}_ROOT}, trying system path.")
      unset(${name}_ROOT)
    else()
      hpx_info("find_library.${name}" "Library found in ${${name}_ROOT}.") 
    endif()
  endif()

  # if not found, retry using system path
  if(NOT ${name}_ROOT)
    find_library(${name}_LIBRARY NAMES ${${name}_LIBRARIES})
    
    if(NOT ${name}_LIBRARY)
      if(${name}_ESSENTIAL)
        hpx_error("find_library.${name}" "Library not found in system path.")
      else() 
        hpx_warn("find_library.${name}" "Library not found in system path.")
      endif()
      unset(${name}_ROOT)
    else()
      hpx_info("find_library.${name}" "Library found in system path.") 
    endif()
  endif()

  set(${name}_FIND_QUIETLY TRUE)
  include(FindPackageHandleStandardArgs)
  find_package_handle_standard_args(${name}
    DEFAULT_MSG ${name}_LIBRARY ${name}_INCLUDE_DIR)

  if(${name}_FOUND)
    get_filename_component(${name}_ROOT ${${name}_INCLUDE_DIR} PATH)
    set(${name}_FOUND ${${name}_FOUND} CACHE BOOL "Found ${name}.")
    set(${name}_ROOT ${${name}_ROOT} CACHE PATH "${name} root directory.")
    set(${name}_LIBRARY ${${name}_LIBRARY} CACHE FILEPATH "${name} shared library.")
    set(${name}_INCLUDE_DIR ${${name}_INCLUDE_DIR} CACHE PATH "${name} include directory.")
  endif()

  endif()
endmacro()

