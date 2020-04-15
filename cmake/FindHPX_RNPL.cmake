# Copyright (c) 2007-2012 Hartmut Kaiser
# Copyright (c) 2010-2011 Matt Anderson
# Copyright (c) 2011      Bryce Lelbach
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

# Set SDF_ROOT in case the other hints are used
if(SDF_ROOT)
  # The call to file is for compatibility with windows paths
  file(TO_CMAKE_PATH ${SDF_ROOT} SDF_ROOT)
elseif("$ENV{SDF_ROOT}")
  file(TO_CMAKE_PATH $ENV{SDF_ROOT} SDF_ROOT)
endif()

if(SDF_ROOT)
  set(RNPL_ROOT "${SDF_ROOT}")
endif()

hpx_find_package(RNPL
  LIBRARIES bbhutil libbbhutil
  LIBRARY_PATHS lib64 lib
  HEADERS bbhutil.h sdf.h
  HEADER_PATHS include)

if(RNPL_FOUND AND NOT HPX_SET_RNPL_MACRO)
  hpx_add_config_define(SDF_FOUND)
  hpx_add_config_define(RNPL_FOUND)
endif()

