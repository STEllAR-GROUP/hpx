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

if(HPX_WITH_PAPI)
  find_package(PAPI)
  if(NOT PAPI_FOUND)
    hpx_error("PAPI could not be found and HPX_WITH_PAPI=On, please specify \
    PAPI_ROOT to point to the root of your PAPI installation")
  endif()
  add_library(hpx::papi INTERFACE IMPORTED)
  set_property(TARGET hpx::papi PROPERTY
    INTERFACE_INCLUDE_DIRECTORIES ${PAPI_INCLUDE_DIR})
  if(${CMAKE_VERSION} VERSION_LESS "3.12.0")
    set_property(TARGET hpx::papi PROPERTY
      INTERFACE_LINK_LIBRARIES ${PAPI_LIBRARY})
  else()
    target_link_libraries(hpx::papi INTERFACE ${PAPI_LIBRARY})
  endif()
  # Construct back HPX_LIBRARIES and HPX_INCLUDE_DIRS to deprecate them progressively
  hpx_include_dirs(${PAPI_INCLUDE_DIR})
  hpx_libraries(${PAPI_LIBRARY})
  ##############################################
endif()
