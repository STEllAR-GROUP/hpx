//  Copyright (c) 2016 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>

namespace hpx { namespace traits { namespace detail {
    // wraps int so that int argument is favored over wrap_int
    struct wrap_int
    {
        constexpr wrap_int(int) {}
    };
}}}    // namespace hpx::traits::detail
