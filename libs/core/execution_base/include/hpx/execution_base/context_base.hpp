//  Copyright (c) 2019 Thomas Heller
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/execution_base/resource_base.hpp>

namespace hpx::execution_base {

    struct context_base
    {
        virtual ~context_base() = default;

        virtual resource_base const& resource() const noexcept = 0;
    };
}    // namespace hpx::execution_base
