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
    namespace lcos {
        namespace detail {
            template <typename Result>
            struct future_data;

            struct future_data_refcnt_base;
        }    // namespace detail

        template <typename R>
        class future;

        template <typename R>
        class shared_future;
    }    // namespace lcos

    using lcos::future;
    using lcos::shared_future;
}    // namespace hpx
