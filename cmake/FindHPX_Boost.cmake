# Copyright (c) 2011 Bryce Lelbach
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

################################################################################
# C++-style include guard to prevent multiple searches in the same build
if(NOT BOOST_SEARCHED)
set(BOOST_SEARCHED ON CACHE INTERNAL "Found Boost libraries")

include(HPX_Utils)

################################################################################
# backwards compatibility
if(BOOST_LIB_DIR AND NOT BOOST_LIBRARY_DIR)
  set(BOOST_LIBRARY_DIR "${BOOST_LIB_DIR}")
endif()

if(NOT BOOST_VERSION_FOUND)
  include(FindHPX_BoostVersion)
endif()

macro(get_boost_compiler_version)
  if(NOT BOOST_COMPILER_VERSION_FOUND)
    set(BOOST_COMPILER_VERSION_FOUND ON CACHE INTERNAL "Boost compiler version was found.")

    if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang"
        OR "${CMAKE_CXX_COMPILER}" MATCHES "clang")
      # clang -dumpversion emulates GCC 4.2.1, instead we use --version
      exec_program(${CMAKE_CXX_COMPILER}
        ARGS ${CMAKE_CXX_COMPILER_ARG1} --version
        OUTPUT_VARIABLE BOOST_COMPILER_VERSION_NUMBER)

      string(REGEX REPLACE "clang version ([0-9])\\.([0-9]) .*" "\\1\\2"
        BOOST_COMPILER_VERSION_NUMBER ${BOOST_COMPILER_VERSION_NUMBER})
    elseif(NOT MSVC)
      exec_program(${CMAKE_CXX_COMPILER}
        ARGS ${CMAKE_CXX_COMPILER_ARG1} -dumpversion
        OUTPUT_VARIABLE BOOST_COMPILER_VERSION_NUMBER)

      string(REGEX REPLACE "([0-9])\\.([0-9])(\\.[0-9])?" "\\1\\2"
        BOOST_COMPILER_VERSION_NUMBER ${BOOST_COMPILER_VERSION_NUMBER})
    endif()
  endif()
endmacro()

macro(build_boost_libname BOOST_RAW_NAME)
  if(NOT BOOST_VERSION_FOUND)
    get_boost_version()
  endif()

  # Attempt to guess the compiler suffix
  if(NOT BOOST_COMPILER_VERSION_FOUND)
    get_boost_compiler_version()
  endif()

  set(BOOST_COMPILER_VERSION "")
  set(BOOST_LIB_SUFFIX "")

  if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel"
      OR "${CMAKE_CXX_COMPILER}" MATCHES "icl"
      OR "${CMAKE_CXX_COMPILER}" MATCHES "icpc")
    if(WIN32)
      set(BOOST_COMPILER_VERSION "-iw")
    else()
      set(BOOST_COMPILER_VERSION "-il")
    endif()
  elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang"
          OR "${CMAKE_CXX_COMPILER}" MATCHES "clang")
    if(APPLE)
        set(BOOST_COMPILER_VERSION "-clang-darwin${BOOST_COMPILER_VERSION_NUMBER}")
    else()
      set(BOOST_COMPILER_VERSION "-clang")
    endif()
  elseif(MSVC12)
    set(BOOST_COMPILER_VERSION "-vc120")
  elseif(MSVC11)
    set(BOOST_COMPILER_VERSION "-vc110")
  elseif(MSVC10)
    set(BOOST_COMPILER_VERSION "-vc100")
  elseif(MSVC90)
    set(BOOST_COMPILER_VERSION "-vc90")
  elseif(MSVC80)
    set(BOOST_COMPILER_VERSION "-vc80")
  elseif(MSVC71)
    set(BOOST_COMPILER_VERSION "-vc71")
  elseif(MSVC70) # Good luck!
    set(BOOST_COMPILER_VERSION "-vc7")
  elseif(MSVC60) # Good luck!
    set(BOOST_COMPILER_VERSION "-vc6")
  elseif(BORLAND)
    set(BOOST_COMPILER_VERSION "-bcb")
  elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "SunPro")
    set(BOOST_COMPILER_VERSION "-sw")
  elseif(MINGW)
    if(${BOOST_MAJOR_VERSION}.${BOOST_MINOR_VERSION} VERSION_LESS 1.34)
      set(BOOST_COMPILER_VERSION "-mgw") # no GCC version encoding prior to 1.34
    else()
      set(BOOST_COMPILER_VERSION "-mgw${BOOST_COMPILER_VERSION_NUMBER}")
    endif()
  elseif(UNIX AND CMAKE_COMPILER_IS_GNUCXX)
    if(${BOOST_MAJOR_VERSION}.${BOOST_MINOR_VERSION} VERSION_LESS 1.34)
      set(BOOST_COMPILER_VERSION "-gcc") # no GCC version encoding prior to 1.34
    else()
      # Determine which version of GCC we have.
      if(APPLE)
        if(BOOST_MINOR_VERSION)
          if(${BOOST_MINOR_VERSION} GREATER 35)
            # In Boost 1.36.0 and newer, the mangled compiler name used
            # on Mac OS X/Darwin is "xgcc".
            set(BOOST_COMPILER_VERSION "-xgcc${BOOST_COMPILER_VERSION_NUMBER}")
          else(${BOOST_MINOR_VERSION} GREATER 35)
            # In Boost <= 1.35.0, there is no mangled compiler name for the Mac OS X/Darwin version of GCC.
            set(BOOST_COMPILER_VERSION "")
          endif(${BOOST_MINOR_VERSION} GREATER 35)
        else(BOOST_MINOR_VERSION)
          # We don't know the Boost version, so assume it's pre-1.36.0.
          set(BOOST_COMPILER_VERSION "")
        endif()
      else()
        set(BOOST_COMPILER_VERSION "-gcc${BOOST_COMPILER_VERSION_NUMBER}")
      endif()
    endif()
  endif()

  set(BOOST_COMPILER_VERSION ${BOOST_COMPILER_VERSION}
    CACHE INTERNAL "Boost compiler version.")

  if(BOOST_SUFFIX)
    # user suffix
    set(BOOST_SUFFIX ${BOOST_SUFFIX} CACHE STRING
      "Boost library suffix (default: `-gd' for Debug builds)." FORCE)
    set(BOOST_LIB_SUFFIX ${BOOST_SUFFIX})
    set(BOOST_FULL_LIB_SUFFIX ${BOOST_LIB_SUFFIX})
  elseif("${CMAKE_BUILD_TYPE}" MATCHES "Debug")
    set(BOOST_LIB_SUFFIX "-gd")
  endif()

  # if BOOST_SUFFIX is specified the user takes full responsibility for it
  if(NOT BOOST_SUFFIX)
    set(BOOST_FULL_LIB_SUFFIX -mt${BOOST_LIB_SUFFIX})
    set(BOOST_LIB_SUFFIX -mt)
  endif()

  set(BOOST_LIBNAMES
      boost_${BOOST_RAW_NAME}${BOOST_COMPILER_VERSION}${BOOST_FULL_LIB_SUFFIX}-${BOOST_VERSION}
      boost_${BOOST_RAW_NAME}${BOOST_COMPILER_VERSION}${BOOST_LIB_SUFFIX}-${BOOST_VERSION}
      boost_${BOOST_RAW_NAME}${BOOST_COMPILER_VERSION}-${BOOST_VERSION}
      boost_${BOOST_RAW_NAME}${BOOST_LIB_SUFFIX}
      boost_${BOOST_RAW_NAME})
  if(NOT MSVC)
    set(BOOST_LIBNAMES ${BOOST_LIBNAMES}
        libboost_${BOOST_RAW_NAME}${BOOST_COMPILER_VERSION}${BOOST_FULL_LIB_SUFFIX}-${BOOST_VERSION}
        libboost_${BOOST_RAW_NAME}${BOOST_COMPILER_VERSION}${BOOST_LIB_SUFFIX}-${BOOST_VERSION}
        libboost_${BOOST_RAW_NAME}${BOOST_COMPILER_VERSION}-${BOOST_VERSION}
        libboost_${BOOST_RAW_NAME}${BOOST_LIB_SUFFIX}
        libboost_${BOOST_RAW_NAME})
  endif()
endmacro()

macro(find_boost_library target_lib)
  if(NOT BOOST_VERSION_FOUND)
    get_boost_version()
  endif()

  string(TOUPPER ${target_lib} target_lib_uc)

  # Non-system Boost tree (tarball, SCM checkout, etc)
  if(NOT BOOST_USE_SYSTEM AND BOOST_LIBRARY_DIR)
    # Locate libraries
    build_boost_libname(${target_lib})
    hpx_print_list("DEBUG" "boost.${target_lib}" "Searching in ${BOOST_LIBRARY_DIR} for" BOOST_LIBNAMES)

    foreach(BOOST_TARGET ${BOOST_LIBNAMES})
      unset(BOOST_${target_lib_uc}_LIBRARY CACHE)      # force the library to be found again
      find_library(BOOST_${target_lib_uc}_LIBRARY NAMES ${BOOST_TARGET} PATHS ${BOOST_LIBRARY_DIR} NO_DEFAULT_PATH)

      if(BOOST_${target_lib_uc}_LIBRARY)
        get_filename_component(path ${BOOST_${target_lib_uc}_LIBRARY} PATH)
        get_filename_component(name ${BOOST_${target_lib_uc}_LIBRARY} NAME)
        hpx_info("boost.${target_lib}" "Found ${name} in ${path}.")
        break()
        hpx_debug("boost.${target_lib}" "Did not find ${BOOST_${target_lib_uc}_LIBRARY}.")
      endif()
    endforeach()

    if(NOT BOOST_${target_lib_uc}_LIBRARY)
      hpx_warn("boost.${target_lib}" "Could not locate Boost ${target_lib} shared library in ${BOOST_LIBRARY_DIR}. Now searching the system path.")
      unset(BOOST_${target_lib_uc}_LIBRARY)

      build_boost_libname(${target_lib})
      hpx_print_list("DEBUG" "boost.${target_lib}" "Searching in system path for" BOOST_LIBNAMES)

      foreach(BOOST_TARGET ${BOOST_LIBNAMES})
        find_library(BOOST_${target_lib_uc}_LIBRARY NAMES ${BOOST_TARGET})

        if(BOOST_${target_lib_uc}_LIBRARY)
          get_filename_component(path ${BOOST_${target_lib_uc}_LIBRARY} PATH)
          get_filename_component(name ${BOOST_${target_lib_uc}_LIBRARY} NAME)
          hpx_info("boost.${target_lib}" "Found ${name} in ${path}.")
          break()
          hpx_debug("boost.${target_lib}" "Did not find ${BOOST_${target_lib_uc}_LIBRARY}.")
        endif()
      endforeach()

      if(NOT BOOST_${target_lib_uc}_LIBRARY)
        hpx_error("boost.${target_lib}" "Failed to locate library in ${BOOST_LIBRARY_DIR} or in the system path.")
        unset(BOOST_${target_lib_uc}_LIBRARY)
      else()
        set(BOOST_${target_lib_uc}_LIBRARY ${BOOST_${target_lib_uc}_LIBRARY}
          CACHE FILEPATH "Boost ${target_lib} shared library." FORCE)
        mark_as_advanced(FORCE BOOST_${target_lib_uc}_LIBRARY)
      endif()
    else()
      set(BOOST_${target_lib_uc}_LIBRARY ${BOOST_${target_lib_uc}_LIBRARY}
        CACHE FILEPATH "Boost ${target_lib} shared library." FORCE)
      mark_as_advanced(FORCE BOOST_${target_lib_uc}_LIBRARY)
    endif()

    list(APPEND BOOST_FOUND_LIBRARIES ${BOOST_${target_lib_uc}_LIBRARY})

  # System Boost installation (deb, rpm, etc)
  else()
    # Locate libraries
    build_boost_libname(${target_lib})
    hpx_print_list("DEBUG" "boost.${target_lib}" "Searching in system path for" BOOST_LIBNAMES)

    foreach(BOOST_TARGET ${BOOST_LIBNAMES})
      unset(BOOST_${target_lib_uc}_LIBRARY CACHE)      # force thelibrary to be found again
      find_library(BOOST_${target_lib_uc}_LIBRARY NAMES ${BOOST_TARGET})

      if(BOOST_${target_lib_uc}_LIBRARY)
        get_filename_component(path ${BOOST_${target_lib_uc}_LIBRARY} PATH)
        get_filename_component(name ${BOOST_${target_lib_uc}_LIBRARY} NAME)
        hpx_info("boost.${target_lib}" "Found ${name} in ${path}.")
        break()
        hpx_debug("boost.${target_lib}" "Did not find ${BOOST_${target_lib_uc}_LIBRARY}.")
      endif()
    endforeach()

    if(NOT BOOST_${target_lib_uc}_LIBRARY)
      hpx_error("boost.${target_lib}" "Failed to locate library in the system path.")
      set(BOOST_FOUND OFF)
      unset(BOOST_${target_lib_uc}_LIBRARY)
    else()
      set(BOOST_${target_lib_uc}_LIBRARY ${BOOST_${target_lib_uc}_LIBRARY}
        CACHE FILEPATH "Boost ${target_lib} shared library." FORCE)
      mark_as_advanced(FORCE BOOST_${target_lib_uc}_LIBRARY)
    endif()

    list(APPEND BOOST_FOUND_LIBRARIES ${BOOST_${target_lib_uc}_LIBRARY})
  endif()
endmacro()

unset(BOOST_FOUND_LIBRARIES CACHE)

foreach(BOOST_LIB ${BOOST_LIBRARIES})
  find_boost_library(${BOOST_LIB})
endforeach()

set(BOOST_FOUND_LIBRARIES ${BOOST_FOUND_LIBRARIES} CACHE STRING "Boost shared libraries found by CMake (default: none).")

################################################################################

endif()

