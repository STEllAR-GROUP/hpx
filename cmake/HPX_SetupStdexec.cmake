#  Copyright (c) 2024 Isidoros Tsaousis-Seiras
#
#  SPDX-License-Identifier: BSL-1.0
#  Distributed under the Boost Software License, Version 1.0. (See accompanying
#  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

if(STDEXEC_ROOT AND NOT Stdexec_ROOT)
  set(Stdexec_ROOT ${STDEXEC_ROOT})
  # remove STDEXEC_ROOT from the cache
  unset(STDEXEC_ROOT CACHE)
endif()

if(Stdexec_ROOT AND HPX_WITH_FETCH_STDEXEC)
  hpx_warn(
    "Both Stdexec_ROOT and HPX_WITH_FETCH_STDEXEC are provided. HPX_WITH_FETCH_STDEXEC will take precedence."
  )
endif()

function(_hpx_patch_stdexec_header header)
  file(READ "${header}" _content)
  set(_patched_content "${_content}")

  # clang-scan-deps used for C++ modules currently rejects apostrophe-separated
  # pp-number literals in stdexec. Remove digit separators in the fetched copy
  # so dependency scanning succeeds.
  set(_previous_content "")
  while(NOT _patched_content STREQUAL _previous_content)
    set(_previous_content "${_patched_content}")
    string(REGEX REPLACE "([0-9])'([0-9])" "\\1\\2" _patched_content
                         "${_patched_content}"
    )
  endwhile()

  if(NOT _content STREQUAL _patched_content)
    file(WRITE "${header}" "${_patched_content}")
  endif()
endfunction()

function(_hpx_patch_stdexec_headers include_dir)
  if(HPX_WITH_CXX_MODULES AND (CMAKE_CXX_COMPILER_ID STREQUAL "Clang"
                               OR CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
  )
    hpx_info("Patching stdexec headers for C++ modules dependency scanning")
    file(GLOB_RECURSE stdexec_headers "${include_dir}/*.hpp")
    foreach(stdexec_header IN LISTS stdexec_headers)
      _hpx_patch_stdexec_header("${stdexec_header}")
    endforeach()
  endif()
endfunction()

if(HPX_WITH_FETCH_STDEXEC)
  hpx_info(
    "HPX_WITH_FETCH_STDEXEC=${HPX_WITH_FETCH_STDEXEC}, Stdexec will be fetched using CMake's FetchContent and installed alongside HPX (HPX_WITH_STDEXEC_TAG=${HPX_WITH_STDEXEC_TAG})"
  )

  include(FetchContent)
  fetchcontent_declare(
    Stdexec
    GIT_REPOSITORY https://github.com/NVIDIA/stdexec.git
    GIT_TAG ${HPX_WITH_STDEXEC_TAG}
  )

  fetchcontent_getproperties(Stdexec)
  if(NOT Stdexec_POPULATED)
    fetchcontent_populate(Stdexec)
  endif()
  set(Stdexec_ROOT ${stdexec_SOURCE_DIR})
  _hpx_patch_stdexec_headers("${stdexec_SOURCE_DIR}/include")

  add_library(Stdexec INTERFACE)
  target_include_directories(
    Stdexec SYSTEM INTERFACE $<BUILD_INTERFACE:${stdexec_SOURCE_DIR}/include>
                             $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  )

  install(
    TARGETS Stdexec
    EXPORT HPXStdexecTarget
    COMPONENT core
  )

  install(
    DIRECTORY ${Stdexec_ROOT}/include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    COMPONENT core
    FILES_MATCHING
    PATTERN "*.hpp"
  )

  export(
    TARGETS Stdexec
    NAMESPACE Stdexec::
    FILE "${CMAKE_CURRENT_BINARY_DIR}/lib/cmake/${HPX_PACKAGE_NAME}/HPXStdexecTarget.cmake"
  )

  install(
    EXPORT HPXStdexecTarget
    NAMESPACE Stdexec::
    FILE HPXStdexecTarget.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${HPX_PACKAGE_NAME}
    COMPONENT cmake
  )

  # TODO: Enforce a single spelling
  add_library(Stdexec::Stdexec ALIAS Stdexec)
  add_library(STDEXEC::stdexec ALIAS Stdexec)
else()
  find_package(Stdexec)

  if(NOT Stdexec_FOUND)
    hpx_error(
      "Stdexec could not be found, please specify Stdexec_ROOT to point to the correct location or enable HPX_WITH_FETCH_STDEXEC"
    )
  elseif(Stdexec_INCLUDE_DIR)
    _hpx_patch_stdexec_headers("${Stdexec_INCLUDE_DIR}")
  endif()
endif()
