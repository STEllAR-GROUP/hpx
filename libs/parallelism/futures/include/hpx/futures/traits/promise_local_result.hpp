//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/type_support/unused.hpp>

namespace hpx { namespace traits {
    ///////////////////////////////////////////////////////////////////////////
    template <typename Result, typename Enable = void>
    struct promise_local_result
    {
        typedef Result type;
    };

    template <>
    struct promise_local_result<util::unused_type>
    {
        typedef void type;
    };
}}    // namespace hpx::traits
