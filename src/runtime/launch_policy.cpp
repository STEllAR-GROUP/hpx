//  Copyright (c) 2007-2017 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#include <hpx/runtime/launch_policy.hpp>
#include <hpx/runtime/serialization/input_archive.hpp>
#include <hpx/runtime/serialization/output_archive.hpp>
#include <hpx/runtime/serialization/serialize.hpp>

namespace hpx
{
    ///////////////////////////////////////////////////////////////////////////
    const detail::async_policy launch::async =
        detail::async_policy{threads::thread_priority_default};
    const detail::fork_policy launch::fork =
        detail::fork_policy{threads::thread_priority_default};
    const detail::sync_policy launch::sync = detail::sync_policy{};
    const detail::deferred_policy launch::deferred = detail::deferred_policy{};
    const detail::apply_policy launch::apply = detail::apply_policy{};

    const detail::select_policy_generator launch::select =
        detail::select_policy_generator{};

    const detail::policy_holder<> launch::all =
        detail::policy_holder<>{detail::launch_policy::all};
    const detail::policy_holder<> launch::sync_policies =
        detail::policy_holder<>{detail::launch_policy::sync_policies};
    const detail::policy_holder<> launch::async_policies =
        detail::policy_holder<>{detail::launch_policy::async_policies};

    ///////////////////////////////////////////////////////////////////////////
    namespace detail
    {
        void policy_holder_base::load(
            serialization::input_archive& ar, unsigned)
        {
            int value = 0;
            ar & value;
            policy_ = static_cast<launch_policy>(value);
            ar & value;
            priority_ = static_cast<threads::thread_priority>(value);
        }

        void policy_holder_base::save(
            serialization::output_archive& ar, unsigned) const
        {
            int value = static_cast<int>(policy_);
            ar & value;
            value = static_cast<int>(priority_);
            ar & value;
        }
    }
}

