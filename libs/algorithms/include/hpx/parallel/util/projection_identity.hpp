//  Copyright (c) 2015-2020 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>

#include <utility>

namespace hpx { namespace parallel { namespace util {
    ///////////////////////////////////////////////////////////////////////////
    struct projection_identity
    {
        template <typename T>
        HPX_HOST_DEVICE HPX_FORCEINLINE constexpr T&& operator()(T&& val) const
        {
            return std::forward<T>(val);
        }
    };
}}}    // namespace hpx::parallel::util
