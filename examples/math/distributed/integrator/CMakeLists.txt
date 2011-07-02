# Copyright (c) 2011 Bryce Lelbach
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying 
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

add_hpx_component(rational64_integrator
  MODULE math
  SOURCES rational64.cpp
  DEPENDENCIES iostreams_component)

add_hpx_component(double_integrator
  MODULE math
  SOURCES double.cpp
  DEPENDENCIES iostreams_component)

