# Copyright (c) 2007-2012 Hartmut Kaiser
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(dependencies
    distributing_factory_component
    ${HDF5_LIBRARY} ${HDF5_CPP_LIBRARY})

add_hpx_component(interpolate1d
  DEPENDENCIES ${dependencies}
  FOLDER "Examples/Interpolate1D"
  AUTOGLOB)

