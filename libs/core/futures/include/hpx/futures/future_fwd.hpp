//  Copyright (c) 2007-2014 Hartmut Kaiser
//  Copyright (c) 2011      Bryce Lelbach
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>

namespace hpx {

    /// \namespace lcos
    namespace lcos::detail {
        template <typename Result>
        struct future_data;

        template <typename Result>
        struct future_data_base;

        struct future_data_refcnt_base;
    }    // namespace lcos::detail

    template <typename R>
    class future;

    template <typename R>
    class shared_future;

    namespace lcos {
        template <typename R>
        using future HPX_DEPRECATED_V(
            1, 8, "hpx::lcos::future is deprecated. Use hpx::future instead.") =
            hpx::future<R>;

        template <typename R>
        using shared_future HPX_DEPRECATED_V(1, 8,
            "hpx::lcos::shared_future is deprecated. Use hpx::shared_future "
            "instead.") = hpx::shared_future<R>;
    }    // namespace lcos
}    // namespace hpx
