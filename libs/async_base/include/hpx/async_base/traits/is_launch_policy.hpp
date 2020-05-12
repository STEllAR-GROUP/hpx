//  Copyright (c) 2007-2016 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/async_base/launch_policy.hpp>
#include <hpx/type_support/decay.hpp>

#include <type_traits>

namespace hpx { namespace traits {
    namespace detail {
        template <typename Policy>
        struct is_launch_policy
          : std::is_base_of<hpx::detail::policy_holder_base, Policy>
        {
        };
    }    // namespace detail

    template <typename Policy>
    struct is_launch_policy
      : detail::is_launch_policy<typename hpx::util::decay<Policy>::type>
    {
    };
}}    // namespace hpx::traits
