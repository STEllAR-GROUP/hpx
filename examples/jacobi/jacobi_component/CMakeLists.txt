# Copyright (c) 2012 Thomas Heller
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

if(HPX_WITH_CXX11_LAMBDAS)
  add_hpx_component(jacobi_component
      AUTOGLOB
      SOURCE_GLOB "*.c*"
      HEADER_GLOB "*.h*"
      ${jacobi_component_FLAGS}
      FOLDER "Examples/Jacobi"
      COMPONENT_DEPENDENCIES iostreams)
endif()
