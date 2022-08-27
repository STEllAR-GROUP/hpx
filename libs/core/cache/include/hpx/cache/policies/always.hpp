//  Copyright (c) 2007-2022 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx::util::cache::policies {

    ///////////////////////////////////////////////////////////////////////////
    template <typename Entry>
    struct always
    {
        constexpr bool operator()(Entry const&) noexcept
        {
            return true;    // always true
        }
    };
}    // namespace hpx::util::cache::policies
