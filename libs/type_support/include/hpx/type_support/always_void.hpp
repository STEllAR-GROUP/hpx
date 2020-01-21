//  Copyright (c) 2013 Agustin Berge
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_UTIL_ALWAYS_VOID_JUN_20_2013_0401AM
#define HPX_UTIL_ALWAYS_VOID_JUN_20_2013_0401AM

namespace hpx { namespace util {
    template <typename... T>
    struct always_void
    {
        typedef void type;
    };
}}    // namespace hpx::util

#endif
