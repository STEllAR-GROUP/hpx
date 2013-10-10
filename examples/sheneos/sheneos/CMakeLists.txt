# Copyright (c) 2007-2012 Hartmut Kaiser
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

find_package(HPX_HDF5)

set(dependencies
    ${HDF5_CPP_LIBRARY} ${HDF5_LIBRARY}
    distributing_factory_component)

add_hpx_component(sheneos
  DEPENDENCIES ${dependencies}
  FOLDER "Examples/ShenEOS"
  AUTOGLOB)

