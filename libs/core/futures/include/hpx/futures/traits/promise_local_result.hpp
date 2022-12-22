//  Copyright (c) 2007-2022 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/type_support/unused.hpp>

namespace hpx::traits {

    ///////////////////////////////////////////////////////////////////////////
    template <typename Result, typename Enable = void>
    struct promise_local_result
    {
        using type = Result;
    };

    template <>
    struct promise_local_result<util::unused_type>
    {
        using type = void;
    };

    template <typename Result>
    using promise_local_result_t = typename promise_local_result<Result>::type;
}    // namespace hpx::traits
