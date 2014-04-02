# Copyright (c) 2011 Bryce Lelbach
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(HPX_GFORTRANVERSION_LOADED TRUE)

if(NOT MSVC)
  include(HPX_Include)

  hpx_include(Compile
              GetIncludeDirectory)

  hpx_get_include_directory(include_dir)

  set(source_dir "")
  if(hpx_SOURCE_DIR)
    set(source_dir "${hpx_SOURCE_DIR}/cmake/tests")
  elseif(HPX_ROOT)
    set(source_dir "${HPX_ROOT}/share/hpx/cmake/tests")
  elseif($ENV{HPX_ROOT})
    set(source_dir "$ENV{HPX_ROOT}/share/hpx/cmake/tests")
  endif()

  file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/${CMAKE_FILES_DIRECTORY}/config_tests")

  hpx_compile(gfortran_version SOURCE "${source_dir}/gfortran_version.fpp"
    LANGUAGE Fortran
    OUTPUT "${CMAKE_BINARY_DIR}/${CMAKE_FILES_DIRECTORY}/config_tests/gfortran_version")

  if("${gfortran_version_RESULT}" STREQUAL "0")
    execute_process(
      COMMAND "${CMAKE_BINARY_DIR}/${CMAKE_FILES_DIRECTORY}/config_tests/gfortran_version"
      OUTPUT_VARIABLE GFORTRAN_VERSION ERROR_QUIET)

    if("${GFORTRAN_VERSION}" STREQUAL "")
      set(GFORTRAN_VERSION "000000" CACHE INTERNAL "" FORCE)
      set(GFORTRAN_VERSION_NUM "000000" CACHE INTERNAL "" FORCE)
      set(GFORTRAN_MAJOR_VERSION "00" CACHE INTERNAL "" FORCE)
      set(GFORTRAN_MINOR_VERSION "00" CACHE INTERNAL "" FORCE)
      set(GFORTRAN_PATCH_VERSION "00" CACHE INTERNAL "" FORCE)
      set(GFORTRAN_VERSION_STR "unknown" CACHE INTERNAL "" FORCE)
    else()
      math(EXPR GFORTRAN_MAJOR_VERSION "${GFORTRAN_VERSION} / 10000")
      math(EXPR GFORTRAN_MINOR_VERSION "${GFORTRAN_VERSION} / 100 % 100")
      math(EXPR GFORTRAN_PATCH_VERSION "${GFORTRAN_VERSION} % 100")

      set(GFORTRAN_VERSION "${GFORTRAN_VERSION}" CACHE INTERNAL "" FORCE)
      set(GFORTRAN_VERSION_NUM "${GFORTRAN_VERSION}" CACHE INTERNAL "" FORCE)
      set(GFORTRAN_MAJOR_VERSION "${GFORTRAN_MAJOR_VERSION}" CACHE INTERNAL "" FORCE)
      set(GFORTRAN_MINOR_VERSION "${GFORTRAN_MINOR_VERSION}" CACHE INTERNAL "" FORCE)
      set(GFORTRAN_PATCH_VERSION "${GFORTRAN_PATCH_VERSION}" CACHE INTERNAL "" FORCE)
      set(GFORTRAN_VERSION_STR
        "${GFORTRAN_MAJOR_VERSION}.${GFORTRAN_MINOR_VERSION}.${GFORTRAN_PATCH_VERSION}"
        CACHE INTERNAL "" FORCE)
    endif()
  else()
    set(GFORTRAN_VERSION "000000" CACHE INTERNAL "" FORCE)
    set(GFORTRAN_VERSION_NUM "000000" CACHE INTERNAL "" FORCE)
    set(GFORTRAN_MAJOR_VERSION "00" CACHE INTERNAL "" FORCE)
    set(GFORTRAN_MINOR_VERSION "00" CACHE INTERNAL "" FORCE)
    set(GFORTRAN_PATCH_VERSION "00" CACHE INTERNAL "" FORCE)
    set(GFORTRAN_VERSION_STR "unknown" CACHE INTERNAL "" FORCE)
  endif()
endif()

