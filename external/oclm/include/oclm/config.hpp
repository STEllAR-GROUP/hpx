
//  Copyright (c) 2012 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef OCLM_CONFIG_HPP
#define OCLM_CONFIG_HPP

#if defined(__APPLE__) || defined(__MACOSX)
//#include <OpenGL/OpenGL.h>
#include <OpenCL/opencl.h>
#else
//#include <GL/gl.h>
#include <CL/opencl.h>
#endif // !__APPLE__

#endif
