# Copyright (c) 2019 Ste||ar Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

# The goal is to store all HPX_* cache variables in a file, so that they would
# be forwarded to projects using HPX (the file is included in the
# HPXConfig.cmake)

function(write_license_header filename)
  file(WRITE ${filename}
"# Copyright (c) 2019 Ste||ar Group\n\
#\n\
# SPDX-License-Identifier: BSL-1.0\n\
# Distributed under the Boost Software License, Version 1.0. (See accompanying\n\
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)\n\n"
    )
endfunction(write_license_header)

get_cmake_property(cache_vars CACHE_VARIABLES)

# Keep only the HPX_* like variables
list(FILTER cache_vars INCLUDE REGEX HPX_)
list(FILTER cache_vars EXCLUDE REGEX "Category$")

# Write the HPXCacheVariables.cmake in the BUILD directory
set(_cache_var_file
  ${CMAKE_CURRENT_BINARY_DIR}/lib/cmake/${HPX_PACKAGE_NAME}/${HPX_PACKAGE_NAME}CacheVariables.cmake)
write_license_header(${_cache_var_file})
file(APPEND ${_cache_var_file} "# File to store the HPX_* cache variables\n")
foreach(_var IN LISTS cache_vars)
  file(APPEND ${_cache_var_file} "set(${_var} ${${_var}})\n")
endforeach()
file(INSTALL ${_cache_var_file}
  DESTINATION ${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY})

# Install the HPXCacheVariables.cmake in the INSTALL directory
install(
  FILES ${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/${HPX_PACKAGE_NAME}CacheVariables.cmake
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${HPX_PACKAGE_NAME}
  COMPONENT cmake
  )

