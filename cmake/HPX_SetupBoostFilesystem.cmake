# Copyright (c) 2019 The STE||AR-Group
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(__filesystem_compatibility_default ON)
if(HPX_WITH_CXX17_FILESYSTEM)
  set(__filesystem_compatibility_default OFF)
endif()

include(HPX_Option)
# Compatibility with using Boost.FileSystem, introduced in V1.4.0
hpx_option(HPX_FILESYSTEM_WITH_BOOST_FILESYSTEM_COMPATIBILITY
  BOOL "Enable Boost.FileSystem compatibility. (default: ${__filesystem_compatibility_default})"
  ${__filesystem_compatibility_default} ADVANCED CATEGORY "Modules")

# Creates imported hpx::boost::filesystem target
if(HPX_FILESYSTEM_WITH_BOOST_FILESYSTEM_COMPATIBILITY)

  hpx_add_config_define_namespace(
    DEFINE HPX_FILESYSTEM_HAVE_BOOST_FILESYSTEM_COMPATIBILITY
    NAMESPACE FILESYSTEM)

  find_package(Boost ${Boost_MINIMUM_VERSION}
    QUIET MODULE
    COMPONENTS filesystem)

  if(NOT Boost_FILESYSTEM_FOUND)
    hpx_error("Could not find Boost.Filesystem. Provide a boost installation \
    including the filesystem library or use a compiler with support for the \
    C++17 filesystem library")
  else()
    hpx_info("    Boost library found: filesystem")
  endif()

  add_library(hpx::boost::filesystem INTERFACE IMPORTED)
  set_property(TARGET hpx::boost::filesystem PROPERTY INTERFACE_LINK_LIBRARIES
    ${Boost_FILESYSTEM_LIBRARIES})

else()
  if(NOT HPX_WITH_CXX17_FILESYSTEM)
    hpx_error("Could not find std::filesystem. Use a compiler with support for "
      "the C++17 filesystem library or enable Boost.FileSystem compatibility "
      "(set HPX_FILESYSTEM_WITH_BOOST_FILESYSTEM_COMPATIBILITY to ON)")
  endif()
endif()
