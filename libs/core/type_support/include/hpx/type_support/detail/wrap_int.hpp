//  Copyright (c) 2016-2023 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

namespace hpx::traits::detail {

    // wraps int so that int argument is favored over wrap_int
    struct wrap_int
    {
        /*implicit*/ constexpr wrap_int(int) noexcept {}
    };
}    // namespace hpx::traits::detail
