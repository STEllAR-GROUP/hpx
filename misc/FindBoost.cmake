# Copyright (c) 2011 Bryce Lelbach
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying 
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

macro(get_boost_version)
  if(NOT BOOST_VERSION_FOUND)
  set(BOOST_VERSION_FOUND ON)

  if(NOT BOOST_USE_SYSTEM)
    if(NOT BOOST_ROOT)
      if(NOT $ENV{BOOST_ROOT} STREQUAL "")
        set(BOOST_ROOT $ENV{BOOST_ROOT})
      elseif(NOT $ENV{BOOST} STREQUAL "")  
        set(BOOST_ROOT $ENV{BOOST})
      else()
        set(BOOST_USE_SYSTEM ON)
      endif()
    endif()
  endif()
  
  #############################################################################
  # Non-system Boost tree (tarball, SCM checkout, etc)
  #############################################################################
  if(BOOST_ROOT)
    if(NOT BOOST_LIB_DIR)
      set(BOOST_LIB_DIR ${BOOST_ROOT}/stage/lib)
    endif()
    
    if(NOT BOOST_INCLUDE_DIR)
      set(BOOST_INCLUDE_DIR ${BOOST_ROOT})
    endif()
  
    ###########################################################################
    # Locate include directory
    ###########################################################################
    find_path(BOOST_VERSION_HPP boost/version.hpp
              PATHS ${BOOST_INCLUDE_DIR} NO_DEFAULT_PATH)
  
    if(NOT BOOST_VERSION_HPP)
      message(WARNING "Could not locate Boost include directory in ${BOOST_INCLUDE_DIR}. Now searching the system.")
      
      find_path(BOOST_VERSION_HPP boost/version.hpp)
    
      if(NOT BOOST_VERSION_HPP)
        message(FATAL_ERRRO "Failed to locate Boost include directory in ${BOOST_INCLUDE_DIR} or in the default path.")
      endif()
    endif()
  
  #############################################################################
  # System Boost installation (deb, rpm, etc)
  #############################################################################
  else()
    ###########################################################################
    # Locate include directory
    ###########################################################################
    find_path(BOOST_VERSION_HPP boost/version.hpp)
    
    if(NOT BOOST_VERSION_HPP)
      message(FATAL_ERROR "Failed to locate Boost include directory in ${BOOST_INCLUDE_DIR} or in the default path.")
    endif()
  
  endif()    
  
  set(BOOST_VERSION_HPP ${BOOST_VERSION_HPP}/boost/version.hpp) 
  message(STATUS "Using ${BOOST_VERSION_HPP} as Boost version.hpp header.")
  
  #############################################################################
  # Get Boost version 
  #############################################################################
  set(BOOST_VERSION "")
  set(BOOST_VERSION_NUM 0)
  set(BOOST_MAJOR_VERSION 0)
  set(BOOST_MINOR_VERSION 0)
  set(BOOST_PATCH_VERSION 0)
  
  file(READ "${BOOST_VERSION_HPP}" BOOST_VERSION_HPP_CONTENTS)
  
  string(REGEX REPLACE ".*#define BOOST_LIB_VERSION \"([0-9_]+)\".*" "\\1"
    BOOST_VERSION "${BOOST_VERSION_HPP_CONTENTS}")
  string(REGEX REPLACE ".*#define BOOST_VERSION ([0-9]+).*" "\\1"
    BOOST_VERSION_NUM "${BOOST_VERSION_HPP_CONTENTS}")
    
  if(NOT "${BOOST_VERSION_NUM}" STREQUAL "0")
    math(EXPR BOOST_MAJOR_VERSION "${BOOST_VERSION_NUM} / 100000")
    math(EXPR BOOST_MINOR_VERSION "${BOOST_VERSION_NUM} / 100 % 1000")
    math(EXPR BOOST_PATCH_VERSION "${BOOST_VERSION_NUM} % 100")
  else()
    message(FATAL_ERROR "Invalid Boost version ${BOOST_VERSION_NUM}.")
  endif()
  
  message(STATUS "Boost version is ${BOOST_MAJOR_VERSION}.${BOOST_MINOR_VERSION}.${BOOST_PATCH_VERSION}.")
  
  set(BOOST_VERSION_HPP ${BOOST_VERSION_HPP}
    CACHE FILEPATH "Boost version.hpp header.")
  set(BOOST_VERSION ${BOOST_VERSION}
    CACHE STRING "Boost version (M.mm.p string version).")
  set(BOOST_VERSION_NUM ${BOOST_VERSION_NUM}
    CACHE STRING "Boost version (unsigned integer version).")
  set(BOOST_MAJOR_VERSION ${BOOST_MAJOR_VERSION}
    CACHE STRING "Boost major version (M).")
  set(BOOST_MINOR_VERSION ${BOOST_MINOR_VERSION}
    CACHE STRING "Boost minor version (mm).")
  set(BOOST_PATCH_VERSION ${BOOST_PATCH_VERSION}
    CACHE STRING "Boost patch version (p).")

  set(BOOST_USE_SYSTEM ${BOOST_USE_SYSTEM} CACHE BOOL
    "Set to true to search for a system install of Boost (default ON).")
  set(BOOST_ROOT ${BOOST_ROOT} CACHE FILEPATH
    "The Boost source tree to use (default: BOOST_ROOT or BOOST environmental variable).")
  set(BOOST_LIB_DIR ${BOOST_LIB_DIR} CACHE FILEPATH
    "Path to Boost shared libraries (default: \${BOOST_ROOT}/stage/lib).")
  set(BOOST_INCLUDE_DIR ${BOOST_INCLUDE_DIR} CACHE FILEPATH
    "Include path for Boost (default: \${BOOST_ROOT}).")
  
  endif()
endmacro()

macro(find_boost_library TARGET_LIB)
  if(NOT BOOST_VERSION_FOUND)
    get_boost_version()
  endif()
 
  #############################################################################
  # Non-system Boost tree (tarball, SCM checkout, etc)
  #############################################################################
  if(NOT BOOST_USE_SYSTEM)
    ###########################################################################
    # Locate libraries 
    ###########################################################################
    find_library(BOOST_${TARGET_LIB}_LIBRARY
        NAMES boost_${TARGET_LIB} PATHS ${BOOST_LIB_DIR} NO_DEFAULT_PATH)
    
    if(NOT BOOST_${TARGET_LIB}_LIBRARY)
      message(WARNING "Could not locate Boost ${TARGET_LIB} shared library in ${BOOST_LIB_DIR}. Now searching the system.")
      unset(BOOST_${TARGET_LIB}_LIBRARY)
      
      find_library(BOOST_${TARGET_LIB}_LIBRARY NAMES boost_${TARGET_LIB})
    
      if(NOT BOOST_${TARGET_LIB}_LIBRARY)
        set(BOOST_ERROR_MSG "Failed to locate Boost ${TARGET_LIB} shared library in ${BOOST_LIB_DIR} or in the default path.")
        message(FATAL_ERROR "${BOOST_ERROR_MSG}")
        unset(BOOST_${TARGET_LIB}_LIBRARY)
      else()
        set(BOOST_${TARGET_LIB}_LIBRARY ${BOOST_${TARGET_LIB}_LIBRARY}
          CACHE FILEPATH "Boost ${TARGET_LIB} shared library.")
        message(STATUS "Using ${BOOST_${TARGET_LIB}_LIBRARY} as Boost ${TARGET_LIB} shared library.")
      endif()
    else()
      set(BOOST_${TARGET_LIB}_LIBRARY ${BOOST_${TARGET_LIB}_LIBRARY}
        CACHE FILEPATH "Boost ${TARGET_LIB} shared library.")
      message(STATUS "Using ${BOOST_${TARGET_LIB}_LIBRARY} as Boost ${TARGET_LIB} shared library.")
    endif()
    
    list(APPEND BOOST_FOUND_LIBRARIES ${BOOST_${TARGET_LIB}_LIBRARY})
  
  #############################################################################
  # System Boost installation (deb, rpm, etc)
  #############################################################################
  else()
    ###########################################################################
    # Locate libraries 
    ###########################################################################
    find_library(BOOST_${TARGET_LIB}_LIBRARY NAMES boost_${TARGET_LIB})
    
    if(NOT BOOST_${TARGET_LIB}_LIBRARY)
      message(FATAL_ERROR "Failed to locate Boost ${TARGET_LIB} shared library in ${BOOST_LIB_DIR} or in the default path.")
      set(BOOST_FOUND OFF)
      unset(BOOST_${TARGET_LIB}_LIBRARY)
    else()
      set(BOOST_${TARGET_LIB}_LIBRARY ${BOOST_${TARGET_LIB}_LIBRARY}
        CACHE FILEPATH "Boost ${TARGET_LIB} shared library.")
      message(STATUS "Using ${BOOST_${TARGET_LIB}_LIBRARY} as Boost ${TARGET_LIB} shared library.")
    endif()
  
    list(APPEND BOOST_FOUND_LIBRARIES ${BOOST_${TARGET_LIB}_LIBRARY})
  endif()    
  
  set(BOOST_FOUND_LIBRARIES ${BOOST_FOUND_LIBRARIES} CACHE STRING
    "Boost shared libraries found by CMake (default: none).")
endmacro()

