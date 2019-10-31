# Copyright (c) 2018 Christopher Hinz
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

if(HPX_WITH_VALGRIND)
  find_package(Valgrind)
  if(NOT VALGRIND_FOUND)
    hpx_error("Valgrind could not be found and HPX_WITH_VALGRIND=On, please \
    specify VALGRIND_ROOT to point to the root of your Valgrind installation")
  endif()

  add_library(hpx::valgrind INTERFACE IMPORTED)
  # FIXME : specify the SYSTEM flag when passing to target_include_directories
  set_property(TARGET hpx::valgrind PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${VALGRIND_INCLUDE_DIR})
  hpx_add_config_define(HPX_HAVE_VALGRIND)
  # Construct back HPX_INCLUDE_DIRS to deprecate them progressively
  hpx_include_dirs(${VALGRIND_INCLUDE_DIR})
  ##############################################
endif()
