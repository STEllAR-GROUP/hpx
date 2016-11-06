# Copyright (c) 2016 Hartmut Kaiser
# Copyright (c) 2016 Andreas Schaefer
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

# Locate LibFlatArray ( http://libgeodecomp.org/libflatarray.html )
#
find_package(libflatarray NO_MODULE PATHS ${LIBFLATARRAY_ROOT})
if(NOT libflatarray_FOUND)
  hpx_error("LibFlatArray was not found while datapar support was requested. Set LIBFLATARRAY_ROOT to the installation path of LibFlatArray")
endif()

include_directories(SYSTEM ${libflatarray_INCLUDE_DIRS})

hpx_add_config_define(HPX_HAVE_DATAPAR)
hpx_add_config_define(HPX_HAVE_DATAPAR_LIBFLATARRAY)

hpx_info("Found libflatarray (vectorization): " ${libflatarray_INCLUDE_DIRS})

