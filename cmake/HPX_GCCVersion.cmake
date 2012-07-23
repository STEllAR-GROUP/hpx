# Copyright (c) 2011 Bryce Lelbach
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(HPX_GCCVERSION_LOADED TRUE)

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

  file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/${CMAKE_FILES_DIRECTORY}/config_tests)

  hpx_compile(gcc_version SOURCE ${source_dir}/gcc_version.cpp
    LANGUAGE CXX
    OUTPUT ${CMAKE_BINARY_DIR}/${CMAKE_FILES_DIRECTORY}/config_tests/gcc_version
    FLAGS -I${BOOST_INCLUDE_DIR} ${include_dir})

  if("${gcc_version_RESULT}" STREQUAL "0")
    if(NOT GCC_VERSION)
      execute_process(
        COMMAND "${CMAKE_BINARY_DIR}/${CMAKE_FILES_DIRECTORY}/config_tests/gcc_version"
        OUTPUT_VARIABLE GCC_VERSION)
    endif()

    if("${GCC_VERSION}" STREQUAL "")
      set(GCC_VERSION "000000" CACHE INTERNAL "" FORCE)
      set(GCC_VERSION_NUM "000000" CACHE INTERNAL "" FORCE)
      set(GCC_MAJOR_VERSION "00" CACHE INTERNAL "" FORCE)
      set(GCC_MINOR_VERSION "00" CACHE INTERNAL "" FORCE)
      set(GCC_PATCH_VERSION "00" CACHE INTERNAL "" FORCE)
      set(GCC_VERSION_STR "unknown" CACHE INTERNAL "" FORCE)
    else()
      math(EXPR GCC_MAJOR_VERSION "${GCC_VERSION} / 10000")
      math(EXPR GCC_MINOR_VERSION "${GCC_VERSION} / 100 % 100")
      math(EXPR GCC_PATCH_VERSION "${GCC_VERSION} % 100")

      set(GCC_VERSION "${GCC_VERSION}" CACHE INTERNAL "" FORCE)
      set(GCC_VERSION_NUM "${GCC_VERSION}" CACHE INTERNAL "" FORCE)
      set(GCC_MAJOR_VERSION "${GCC_MAJOR_VERSION}" CACHE INTERNAL "" FORCE)
      set(GCC_MINOR_VERSION "${GCC_MINOR_VERSION}" CACHE INTERNAL "" FORCE)
      set(GCC_PATCH_VERSION "${GCC_PATCH_VERSION}" CACHE INTERNAL "" FORCE)
      set(GCC_VERSION_STR
        "${GCC_MAJOR_VERSION}.${GCC_MINOR_VERSION}.${GCC_PATCH_VERSION}"
        CACHE INTERNAL "" FORCE)
    endif()
  else()
    set(GCC_VERSION "000000" CACHE INTERNAL "" FORCE)
    set(GCC_VERSION_NUM "000000" CACHE INTERNAL "" FORCE)
    set(GCC_MAJOR_VERSION "00" CACHE INTERNAL "" FORCE)
    set(GCC_MINOR_VERSION "00" CACHE INTERNAL "" FORCE)
    set(GCC_PATCH_VERSION "00" CACHE INTERNAL "" FORCE)
    set(GCC_VERSION_STR "unknown" CACHE INTERNAL "" FORCE)
  endif()
endif()

