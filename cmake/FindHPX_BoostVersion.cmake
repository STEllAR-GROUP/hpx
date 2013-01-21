# Copyright (c) 2011 Bryce Lelbach
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

################################################################################
# C++-style include guard to prevent multiple searches in the same build
if(NOT BOOST_VERSION_SEARCHED)

include(HPX_Utils)

################################################################################
macro(get_boost_version)
  if(NOT BOOST_VERSION_FOUND)
  set(BOOST_VERSION_FOUND OFF CACHE INTERNAL "Boost version was found.")

  # Non-system Boost tree (tarball, SCM checkout, etc)
  if(NOT BOOST_USE_SYSTEM)
    if(NOT BOOST_ROOT)
      if(NOT $ENV{BOOST_ROOT} STREQUAL "")
        set(BOOST_ROOT $ENV{BOOST_ROOT})
      elseif(NOT $ENV{BOOST} STREQUAL "")
        set(BOOST_ROOT $ENV{BOOST})
      endif()
    endif()

    if(NOT BOOST_LIBRARY_DIR AND BOOST_ROOT)
      find_path(BOOST_LIBRARY_DIR lib PATHS ${BOOST_ROOT}/stage64 ${BOOST_ROOT}/stage ${BOOST_ROOT} NO_DEFAULT_PATH)
      if(${BOOST_LIBRARY_DIR} STREQUAL BOOST_LIBRARY_DIR-NOTFOUND)
        find_path(BOOST_LIBRARY_DIR lib64 PATHS ${BOOST_ROOT}/stage64 ${BOOST_ROOT}/stage ${BOOST_ROOT} NO_DEFAULT_PATH)
        if(${BOOST_LIBRARY_DIR} STREQUAL BOOST_LIBRARY_DIR-NOTFOUND)
          set(BOOST_LIBRARY_DIR "${BOOST_LIBRARY_DIR}/lib64" )
        endif()
      else()
        set(BOOST_LIBRARY_DIR "${BOOST_LIBRARY_DIR}/lib")
      endif()
      hpx_debug("boost.version" "Using ${BOOST_LIBRARY_DIR} as Boost shared library directory")
    endif()

    if(NOT BOOST_INCLUDE_DIR AND BOOST_ROOT)
      find_path(BOOST_INCLUDE_DIR boost PATHS ${BOOST_ROOT} ${BOOST_ROOT}/include NO_DEFAULT_PATH)
    endif()

    # Locate include directory
    find_path(BOOST_VERSION_HPP boost/version.hpp PATHS ${BOOST_INCLUDE_DIR} NO_DEFAULT_PATH)

    if(${BOOST_VERSION_HPP} STREQUAL BOOST_VERSION_HPP-NOTFOUND)
      set(boost_possible_suffixes
        "boost-1_54" "boost-1_53" "boost-1_52" "boost-1_51" "boost-1_49" "boost-1_48"
        "boost-1_47" "boost-1_46" "boost-1_45" "boost-1.44" "boost-1_43" "boost-1_42")

      if(NOT BOOST_INCLUDE_DIR)
        hpx_warn("boost.version" "Could not locate Boost include directory. Now searching versioned include directory.")
      else()
        hpx_warn("boost.version" "Could not locate Boost include directory in ${BOOST_INCLUDE_DIR}. Now searching versioned include directory.")
      endif()

      find_path(BOOST_VERSION_HPP boost/version.hpp
          PATHS ${BOOST_LIBRARY_DIR}
          PATH_SUFFIXES ${boost_possible_suffixes}
          NO_DEFAULT_PATH)
    endif()

    if(${BOOST_VERSION_HPP} STREQUAL BOOST_VERSION_HPP-NOTFOUND)
      if(NOT BOOST_INCLUDE_DIR)
        hpx_warn("boost.version" "Could not locate Boost include directory. Now searching the system.")
      else()
        hpx_warn("boost.version" "Could not locate Boost include directory in ${BOOST_INCLUDE_DIR}. Now searching the system.")
      endif()

      find_path(BOOST_VERSION_HPP boost/version.hpp)

      if(${BOOST_VERSION_HPP} STREQUAL BOOST_VERSION_HPP-NOTFOUND)
        if(NOT BOOST_INCLUDE_DIR)
          hpx_error("boost.version" "Failed to locate Boost include directory in the default path. Try setting BOOST_ROOT to your Boost installation directory.")
        else()
          hpx_error("boost.version" "Failed to locate Boost include directory in ${BOOST_INCLUDE_DIR} or in the default path. Try setting BOOST_ROOT to your Boost installation directory.")
        endif()
      endif()
    endif()

  # System Boost installation (deb, rpm, etc)
  else()
    # Locate include directory
    find_path(BOOST_VERSION_HPP boost/version.hpp)

    if(${BOOST_VERSION_HPP} STREQUAL BOOST_VERSION_HPP-NOTFOUND)
      hpx_error("boost.version" "Failed to locate Boost include directory in the default path. Try setting BOOST_ROOT to your Boost installation directory.")
    endif()

  endif()

  set(BOOST_VERSION_HPP ${BOOST_VERSION_HPP}/boost/version.hpp)
  hpx_info("boost.version" "Using ${BOOST_VERSION_HPP} as Boost version.hpp header.")

  # Get Boost version
  set(BOOST_VERSION "")
  set(BOOST_VERSION_STR "")
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
    hpx_error("boost.version" "Invalid Boost version ${BOOST_VERSION_NUM}, expected format MMRRPP (i.e. 104800)")
  endif()

  set(BOOST_VERSION_HPP "${BOOST_VERSION_HPP}"
    CACHE FILEPATH "Boost version.hpp header." FORCE)
  set(BOOST_VERSION "${BOOST_VERSION}"
    CACHE STRING "Boost version (M_mm string version)." FORCE)
  set(BOOST_VERSION_NUM "${BOOST_VERSION_NUM}"
    CACHE STRING "Boost version (unsigned integer version)." FORCE)
  set(BOOST_MAJOR_VERSION "${BOOST_MAJOR_VERSION}"
    CACHE STRING "Boost major version (M)." FORCE)
  set(BOOST_MINOR_VERSION "${BOOST_MINOR_VERSION}"
    CACHE STRING "Boost minor version (mm)." FORCE)
  set(BOOST_PATCH_VERSION "${BOOST_PATCH_VERSION}"
    CACHE STRING "Boost patch version (p)." FORCE)
  mark_as_advanced(FORCE BOOST_VERSION_HPP BOOST_VERSION BOOST_VERSION_NUM
    BOOST_MAJOR_VERSION BOOST_MINOR_VERSION BOOST_PATCH_VERSION)

  set(BOOST_VERSION_STR
    "${BOOST_MAJOR_VERSION}.${BOOST_MINOR_VERSION}.${BOOST_PATCH_VERSION}"
    CACHE STRING "Boost version (M.mm.p string version)." FORCE)

  hpx_info("boost.version" "Boost version is ${BOOST_VERSION_STR}.")

  set(BOOST_VERSION_SEARCHED ON CACHE INTERNAL "Found Boost version")
  set(BOOST_VERSION_FOUND ON CACHE INTERNAL "Boost version was found.")

  set(BOOST_USE_SYSTEM "${BOOST_USE_SYSTEM}" CACHE BOOL
    "Set to true to search for a system install of Boost (default ON)." FORCE)
  set(BOOST_ROOT "${BOOST_ROOT}" CACHE PATH
    "The Boost source tree to use (default: BOOST_ROOT or BOOST environmental variable)." FORCE)
  set(BOOST_LIBRARY_DIR "${BOOST_LIBRARY_DIR}" CACHE PATH
    "Path to Boost shared libraries (default: \${BOOST_ROOT}/stage/lib)." FORCE)
  set(BOOST_INCLUDE_DIR "${BOOST_INCLUDE_DIR}" CACHE PATH
    "Include path for Boost (default: \${BOOST_ROOT})." FORCE)
  mark_as_advanced(FORCE BOOST_USE_SYSTEM BOOST_USE_SYSTEM BOOST_LIBRARY_DIR BOOST_INCLUDE_DIR)

  endif()
endmacro()

get_boost_version()

################################################################################

endif()

