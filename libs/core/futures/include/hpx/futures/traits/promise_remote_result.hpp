//  Copyright (c) 2007-2022 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/type_support/unused.hpp>

namespace hpx::traits {

    template <typename Result, typename Enable = void>
    struct promise_remote_result
    {
        using type = Result;
    };

    template <>
    struct promise_remote_result<void>
    {
        using type = hpx::util::unused_type;
    };

    template <typename Result>
    using promise_remote_result_t =
        typename promise_remote_result<Result>::type;
}    // namespace hpx::traits
