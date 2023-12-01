# Copyright (c) 2007-2019 Hartmut Kaiser
# Copyright (c) 2011-2014 Thomas Heller
# Copyright (c) 2007-2008 Chirag Dekate
# Copyright (c)      2011 Bryce Lelbach
# Copyright (c)      2011 Vinay C Amatya
# Copyright (c)      2013 Jeroen Habraken
# Copyright (c) 2014-2016 Andreas Schaefer
# Copyright (c) 2017      Abhimanyu Rawat
# Copyright (c) 2017      Google
# Copyright (c) 2017      Taeguk Kwon
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

if (NOT HPX_WITH_FETCH_HWLOC)
  find_package(Hwloc)
  if(NOT Hwloc_FOUND)
    hpx_error(
      "Hwloc could not be found, please specify Hwloc_ROOT to point to the correct location"
    )
  endif()
else()
  if(UNIX)
    include(FetchContent)
    FetchContent_Declare(HWLoc
      URL https://download.open-mpi.org/release/hwloc/v2.9/hwloc-2.9.3.tar.gz
      TLS_VERIFY true
    )
    if(NOT HWLoc_POPULATED)
      FetchContent_Populate(HWLoc)
      execute_process(COMMAND sh -c "cd ${CMAKE_BINARY_DIR}/_deps/hwloc-src && ./configure --prefix=${CMAKE_BINARY_DIR}/_deps/hwloc-installed && make -j && make install")
    endif()
    set(HWLOC_ROOT "${CMAKE_BINARY_DIR}/_deps/hwloc-installed")
  elseif("${CMAKE_GENERATOR_PLATFORM}" STREQUAL "Win64")
    FetchContent_Declare(HWLoc
      URL https://download.open-mpi.org/release/hwloc/v2.9/hwloc-win64-build-2.9.3.zip
      TLS_VERIFY true
    )
    if(NOT HWLoc_POPULATED)
      FetchContent_Populate(HWLoc)
    endif()
    set(HWLOC_ROOT "${CMAKE_BINARY_DIR}/_deps/hwloc-src" CACHE INTERNAL "")
    find_package(hwloc REQUIRED PATHS ${HWLOC_ROOT} NO_DEFAULT_PATH)
    include_directories(${HWLOC_ROOT}/include)
    link_directories(${HWLOC_ROOT}/lib)
    set(Hwloc_INCLUDE_DIR ${HWLOC_ROOT}/include CACHE INTERNAL "")
    set(Hwloc_LIBRARY ${HWLOC_ROOT}/lib CACHE INTERNAL "")
  else()
    FetchContent_Declare(HWLoc
      URL https://download.open-mpi.org/release/hwloc/v2.9/hwloc-win64-build-2.9.3.zip
      TLS_VERIFY true
    )
    if(NOT HWLoc_POPULATED)
      FetchContent_Populate(HWLoc)
    endif()
    set(HWLOC_ROOT "${CMAKE_BINARY_DIR}/_deps/hwloc-src" CACHE INTERNAL "")
    include_directories(${HWLOC_ROOT}/include)
    link_directories(${HWLOC_ROOT}/lib)
    set(Hwloc_INCLUDE_DIR ${HWLOC_ROOT}/include CACHE INTERNAL "")
    set(Hwloc_LIBRARY ${HWLOC_ROOT}/lib CACHE INTERNAL "")
  endif() # End hwloc installation

  add_library(Hwloc::hwloc INTERFACE IMPORTED)
  target_include_directories(Hwloc::hwloc INTERFACE ${Hwloc_INCLUDE_DIR})
  target_link_libraries(Hwloc::hwloc INTERFACE ${Hwloc_LIBRARY})

endif()
