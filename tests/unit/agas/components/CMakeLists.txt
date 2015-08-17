# Copyright (c) 2007-2012 Hartmut Kaiser
# Copyright (c)      2011 Bryce Lelbach
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(root "${PROJECT_SOURCE_DIR}/tests/unit/agas/components")

add_hpx_component(simple_refcnt_checker
  DEPENDENCIES iostreams_component
  HEADER_GLOB "${root}/simple_refcnt_checker.h*"
  SOURCE_GLOB "${root}/simple_refcnt_checker.c*"
  FOLDER "Tests/Unit/AGAS"
  EXCLUDE_FROM_ALL
  AUTOGLOB)

add_hpx_component(managed_refcnt_checker
  DEPENDENCIES iostreams_component
  HEADER_GLOB "${root}/managed_refcnt_checker.h*"
  SOURCE_GLOB "${root}/managed_refcnt_checker.c*"
  FOLDER "Tests/Unit/AGAS"
  EXCLUDE_FROM_ALL
  AUTOGLOB)

add_hpx_component(simple_mobile_object
  HEADER_GLOB "${root}/simple_mobile_object.h*"
  SOURCE_GLOB "${root}/simple_mobile_object.c*"
  FOLDER "Tests/Unit/AGAS"
  EXCLUDE_FROM_ALL
  AUTOGLOB)

