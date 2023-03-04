//  Copyright (c) 2023 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>

#if defined(HPX_HAVE_CXX20_STD_DEFAULT_SENTINEL)

#include <iterator>

namespace hpx {

    using std::default_sentinel;
    using std::default_sentinel_t;
}    // namespace hpx

#else

namespace hpx {

    struct default_sentinel_t
    {
    };
    inline constexpr default_sentinel_t default_sentinel = default_sentinel_t{};
}    // namespace hpx

#endif
