# Copyright (c) 2012 Thomas Heller
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

if(HPX_HAVE_CXX11_LAMBDAS)

  set(jacobi_component_FLAGS
      DEPENDENCIES iostreams_component
                      distributing_factory_component
                      remote_object_component)

  add_hpx_component(jacobi
      AUTOGLOB
      SOURCE_GLOB "*.c*"
      HEADER_GLOB "*.h*"
      ${jacobi_component_FLAGS}
      FOLDER "Examples/Jacobi")
endif()
