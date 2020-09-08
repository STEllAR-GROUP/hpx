//  Copyright (c) 2016 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <type_traits>

namespace hpx { namespace util {
    template <bool Enable, typename C1, typename C2>
    struct lazy_conditional : std::conditional<Enable, C1, C2>::type
    {
    };
}}    // namespace hpx::util
