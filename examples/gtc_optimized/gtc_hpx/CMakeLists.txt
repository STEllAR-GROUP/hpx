# Copyright (c) 2011 Matt Anderson
# Copyright (c) 2007-2012 Hartmut Kaiser
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

add_hpx_component(gtc_hpx
  SOURCES ${sources}
  DEPENDENCIES iostreams_component
  FOLDER "Examples/GTC Optimized"
  AUTOGLOB)

