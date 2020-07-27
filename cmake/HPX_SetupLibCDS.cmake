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
# Copyright (c) 2018 Christopher Hinz
# Copyright (c) 2020      Weile Wei
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

if(HPX_WITH_LIBCDS AND NOT TARGET LibCDS::cds)
  include(FetchContent)
  include(HPX_Message)

  set(LIBCDS_WITH_HPX
      ON
      CACHE INTERNAL ""
  )
  set(LIBCDS_INSIDE_HPX
      ON
      CACHE INTERNAL ""
  )

  hpx_info(
    "Fetching libCDS from repository: ${HPX_WITH_LIBCDS_GIT_REPOSITORY}, "
    "tag: ${HPX_WITH_LIBCDS_GIT_TAG}"
  )
  fetchcontent_declare(
    libcds
    GIT_REPOSITORY ${HPX_WITH_LIBCDS_GIT_REPOSITORY}
    GIT_TAG ${HPX_WITH_LIBCDS_GIT_TAG}
    GIT_SHALLOW TRUE
  )
  fetchcontent_getproperties(libcds)

  if(NOT libcds_POPULATED)
    fetchcontent_populate(libcds)
    set(LIBCDS_CXX_STANDARD ${HPX_CXX_STANDARD})
    add_subdirectory(${libcds_SOURCE_DIR} ${libcds_BINARY_DIR})

    set_target_properties(cds PROPERTIES FOLDER "Core")
    set_target_properties(cds-s PROPERTIES FOLDER "Core")
  endif()

endif()
