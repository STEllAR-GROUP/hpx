# Copyright (c) 2012 Andrew Kemp
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

if(NOT HPX_FINDPACKAGE_LOADED)
  hpx_include(FindPackage)
endif()

hpx_find_package(OpenCL
  LIBRARIES OpenCL.lib cl libcl OpenCL libOpenCL
  LIBRARY_PATHS lib/x86_64 lib/x64
  HEADERS CL/opencl.h
  HEADER_PATHS include inc)
