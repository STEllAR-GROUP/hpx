# Copyright (c) 2007-2009 Hartmut Kaiser
# Copyright (c)      2011 Bryce Lelbach
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

add_hpx_component(simple_refcnt_checker
  SOURCES simple_refcnt_checker.cpp
          server/simple_refcnt_checker.cpp
  DEPENDENCIES iostreams_component)

add_hpx_component(managed_refcnt_checker
  SOURCES managed_refcnt_checker.cpp
          server/managed_refcnt_checker.cpp
  DEPENDENCIES iostreams_component)

