//  Copyright (c) 2007-2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#include <hpx/runtime/launch_policy.hpp>

namespace hpx
{
    const detail::async_policy launch::async = detail::async_policy{};
    const detail::fork_policy launch::fork = detail::fork_policy{};
    const detail::sync_policy launch::sync = detail::sync_policy{};
    const detail::deferred_policy launch::deferred = detail::deferred_policy{};
    const detail::apply_policy launch::apply = detail::apply_policy{};

    const detail::policy_holder launch::all =
        detail::policy_holder{detail::launch_policy::all};
    const detail::policy_holder launch::sync_policies =
        detail::policy_holder{detail::launch_policy::sync_policies};
    const detail::policy_holder launch::async_policies =
        detail::policy_holder{detail::launch_policy::async_policies};
}

