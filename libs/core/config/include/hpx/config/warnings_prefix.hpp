//  Copyright (c) 2007-2022 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// This file can be included multiple times and does not use #pragma once.
// hpxinspect:nopragmaonce

#include <hpx/config/compiler_specific.hpp>

// suppress warnings about dependent classes not being exported from the dll
#if defined(HPX_MSVC_WARNING_PRAGMA)
#pragma warning(push)
#pragma warning(disable : 4251 4231 4275 4355 4660)
#pragma warning(disable : 4355)    // this used in base member initializer
#endif
