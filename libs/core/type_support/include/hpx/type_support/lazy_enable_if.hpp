//  Copyright (c) 2016 Thomas Heller
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

namespace hpx::util {

    template <bool Enable, typename T>
    struct lazy_enable_if
    {
    };

    template <typename T>
    struct lazy_enable_if<true, T>
    {
        using type = typename T::type;
    };
}    // namespace hpx::util
