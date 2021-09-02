//  Copyright (c) 2007-2018 Hartmut Kaiser
//  Copyright (c) 2011 Bryce Lelbach
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#if defined(DOXYGEN)
/// Defined if HPX is compiled in debug mode.
#define HPX_DEBUG
/// Evaluates to ``debug`` if compiled in debug mode, ``release`` otherwise.
#define HPX_BUILD_TYPE
#else

#if defined(HPX_DEBUG)
#define HPX_BUILD_TYPE debug
#else
#define HPX_BUILD_TYPE release
#endif
#endif
