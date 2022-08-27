# Copyright (c) 2019 Ste||ar Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

# The goal is to store all HPX_* cache variables in a file, so that they would
# be forwarded to projects using HPX (the file is included in the
# HPXConfig.cmake)

get_cmake_property(cache_vars CACHE_VARIABLES)

# Keep only the HPX_* like variables
list(FILTER cache_vars INCLUDE REGEX HPX_)
list(FILTER cache_vars EXCLUDE REGEX "Category$")

# Generate HPXCacheVariables.cmake in the BUILD directory
set(_cache_var_file
    ${CMAKE_CURRENT_BINARY_DIR}/lib/cmake/${HPX_PACKAGE_NAME}/${HPX_PACKAGE_NAME}CacheVariables.cmake
)
set(_cache_var_file_template
    "${HPX_SOURCE_DIR}/cmake/templates/${HPX_PACKAGE_NAME}CacheVariables.cmake.in"
)
set(_cache_variables)
foreach(_var IN LISTS cache_vars)
  if(HPX_WITH_VCPKG)
    # avoid writing directory names into cache file
    string(FIND ${_var} "_DIR" _pos)
    if(NOT ${_pos} EQUAL -1)
      continue()
    endif()
    string(FIND ${_var} "_PATH" _pos)
    if(NOT ${_pos} EQUAL -1)
      continue()
    endif()
  endif()
  set(_cache_variables "${_cache_variables}set(${_var} ${${_var}})\n")
endforeach()

configure_file(${_cache_var_file_template} ${_cache_var_file})

# Install the HPXCacheVariables.cmake in the INSTALL directory
install(
  FILES ${_cache_var_file}
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${HPX_PACKAGE_NAME}
  COMPONENT cmake
)
