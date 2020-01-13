# Copyright (c) 2019 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

# Creates imported hpx::boost::filesystem target
if(HPX_FILESYSTEM_WITH_BOOST_FILESYSTEM_COMPATIBILITY)

  # In case find_package(HPX) is called multiple times
  if (NOT TARGET hpx::boost::filesystem)

    hpx_add_config_define_namespace(
      DEFINE HPX_FILESYSTEM_HAVE_BOOST_FILESYSTEM_COMPATIBILITY
      NAMESPACE FILESYSTEM)

    find_package(Boost ${Boost_MINIMUM_VERSION}
      QUIET MODULE
      COMPONENTS filesystem)

    add_library(hpx::boost::filesystem INTERFACE IMPORTED)

    set_property(TARGET hpx::boost::filesystem APPEND PROPERTY
      INTERFACE_INCLUDE_DIRECTORIES ${Boost_INCLUDE_DIRS})
    if(${CMAKE_VERSION} VERSION_LESS "3.12.0")
      set_property(TARGET hpx::boost::filesystem PROPERTY INTERFACE_LINK_LIBRARIES
        ${Boost_FILESYSTEM_LIBRARIES})
    else()
      target_link_libraries(hpx::boost::filesystem INTERFACE
        ${Boost_FILESYSTEM_LIBRARIES})
    endif()

  endif()

else()
  if(NOT HPX_WITH_CXX17_FILESYSTEM)
    hpx_error("Could not find std::filesystem. Use a compiler with support for "
      "the C++17 filesystem library or enable Boost.FileSystem compatibility "
      "(set HPX_FILESYSTEM_WITH_BOOST_FILESYSTEM_COMPATIBILITY to ON)")
  endif()
endif()
