//  Copyright (c) 2015 Thomas Heller
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

//#if !defined(HPX_HAVE_CXX20_MODULES)
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__)
#define _WINSOCKAPI_
#include <winsock2.h>
#endif
//#endif
