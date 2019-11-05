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

if (NOT TARGET hpx::hwloc)

  find_package(Hwloc)
  if(NOT HWLOC_FOUND)
    hpx_error("Hwloc could not be found, please specify HWLOC_ROOT to point to the correct location")
  endif()

  add_library(hpx::hwloc INTERFACE IMPORTED)
  # System has been removed when passing at set_property for cmake < 3.11
  # instead of target_include_directories
  set_property(TARGET hpx::hwloc PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${HWLOC_INCLUDE_DIR})
  if(${CMAKE_VERSION} VERSION_LESS "3.12.0")
    set_property(TARGET hpx::hwloc PROPERTY INTERFACE_LINK_LIBRARIES ${HWLOC_LIBRARIES})
  else()
    target_link_libraries(hpx::hwloc INTERFACE ${HWLOC_LIBRARIES})
  endif()

endif()
