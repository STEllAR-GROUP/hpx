//  Copyright (c) 2014 Bibek Ghimire
//  Copyright (c) 2014-2015 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>

#include <cstddef>
#include <type_traits>

namespace hpx { namespace traits
{
    template <typename T, typename Enable = void>
    struct is_distribution_policy
      : std::false_type
    {};

    // By default the number of partitions is the same as the number of
    // localities represented by the given distribution policy
    template <typename Policy, typename Enable = void>
    struct num_container_partitions
    {
        static std::size_t call(Policy const& policy)
        {
            return policy.get_num_localities();
        }
    };
}}

