# Copyright (c) 2009-2011 Matthew Anderson
# Copyright (c) 2007-2012 Hartmut Kaiser
# Copyright (c) 2011      Bryce Lelbach
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

find_package(HPX_RNPL)

add_hpx_component(dataflow_c
  DEPENDENCIES dataflow_c_test_lib dataflow_mesh_component
  FOLDER "Examples/Applications/AMR"
  AUTOGLOB)

