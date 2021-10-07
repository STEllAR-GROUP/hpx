# Copyright (c) 2019 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

if(HPXLocal_FILESYSTEM_WITH_BOOST_FILESYSTEM_COMPATIBILITY)
  # In case find_package(HPX) is called multiple times
  if(NOT TARGET Boost::filesystem)
    find_package(Boost ${Boost_MINIMUM_VERSION} MODULE COMPONENTS filesystem)

    if(NOT Boost_FILESYSTEM_FOUND)
      hpx_local_error(
        "Could not find Boost.Filesystem Provide a boost installation including the filesystem library"
      )
    endif()
  endif()

  hpx_local_add_config_define_namespace(
    DEFINE HPX_FILESYSTEM_HAVE_BOOST_FILESYSTEM_COMPATIBILITY
    NAMESPACE FILESYSTEM
  )
else()
  if(NOT HPXLocal_WITH_CXX17_FILESYSTEM)
    hpx_local_error(
      "Could not find std::filesystem. Use a compiler with support for "
      "the C++17 filesystem library or enable Boost.FileSystem compatibility "
      "(set HPXLocal_FILESYSTEM_WITH_BOOST_FILESYSTEM_COMPATIBILITY to ON)"
    )
  endif()
endif()
