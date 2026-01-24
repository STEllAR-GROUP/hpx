// Copyright (c) 2025 Ujjwal Shakeher
// SPDX-License-Identifier: BSL-1.0
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once    // currently only on Clang p2996 fork

#include <hpx/config/defines.hpp>

#if (defined(__clang__) && __clang_major__ >= 20) && !defined(_MSC_VER) &&     \
    !defined(HPX_MINGW)
#define HPX_HAVE_CXX26_EXPERIMENTAL_META

// By default allow auto generation of serialization functions using C++26 reflection
#if !defined(HPX_SERIALIZATION_HAVE_ALLOW_AUTO_GENERATE)
#define HPX_SERIALIZATION_HAVE_ALLOW_AUTO_GENERATE
#endif

#endif
