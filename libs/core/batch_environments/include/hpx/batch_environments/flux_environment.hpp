//  Copyright (c) 2007-2025 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>

#include <cstddef>
#include <string>

namespace hpx::util::batch_environments {

    struct flux_environment
    {
        HPX_CORE_EXPORT flux_environment();

        constexpr bool valid() const noexcept
        {
            return valid_;
        }

        constexpr std::size_t node_num() const noexcept
        {
            return node_num_;
        }

        constexpr std::size_t num_localities() const noexcept
        {
            return num_localities_;
        }

    private:
        std::size_t node_num_;
        std::size_t num_localities_;
        bool valid_;
    };
}    // namespace hpx::util::batch_environments
