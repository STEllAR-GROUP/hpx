//  Copyright (c) 2013 Agustin Berge
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

namespace hpx { namespace util {
    template <typename... T>
    struct always_void
    {
        using type = void;
    };

    template <typename... T>
    using always_void_t = typename always_void<T...>::type;
}}    // namespace hpx::util
