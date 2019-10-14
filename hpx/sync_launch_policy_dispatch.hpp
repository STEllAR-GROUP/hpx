//  Copyright (c) 2007-2017 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_SYNC_LAUNCH_POLICY_DISPATCH_FEB_14_2019_1252PM)
#define HPX_SYNC_LAUNCH_POLICY_DISPATCH_FEB_14_2019_1252PM

#include <hpx/config.hpp>
#include <hpx/lcos/sync_fwd.hpp>
#include <hpx/lcos/future.hpp>
#include <hpx/lcos/local/futures_factory.hpp>
#include <hpx/runtime/launch_policy.hpp>
#include <hpx/functional/traits/is_action.hpp>
#include <hpx/functional/invoke.hpp>

#include <functional>
#include <type_traits>
#include <utility>

namespace hpx { namespace detail
{
    ///////////////////////////////////////////////////////////////////////////
    template <typename Action>
    struct sync_launch_policy_dispatch<Action,
        typename std::enable_if<
            !traits::is_action<Action>::value
        >::type>
    {
        // general case execute on separate thread (except launch::sync)
        template <typename F, typename ...Ts>
        HPX_FORCEINLINE static
        typename hpx::util::detail::invoke_deferred_result<F, Ts...>::type
        call(launch policy, F && f, Ts&&... ts)
        {
            typedef typename util::detail::invoke_deferred_result<F, Ts...>::type
                result_type;

            if (policy == launch::sync)
            {
                return call(
                    launch::sync, std::forward<F>(f), std::forward<Ts>(ts)...);
            }

            lcos::local::futures_factory<result_type()> p(
                util::deferred_call(std::forward<F>(f), std::forward<Ts>(ts)...));

            if (hpx::detail::has_async_policy(policy))
            {
                threads::thread_id_type tid = p.apply(policy, policy.priority());
                if (tid && policy == launch::fork)
                {
                    // make sure this thread is executed last
                    // yield_to
                    hpx::this_thread::suspend(threads::pending, tid,
                        "sync_launch_policy_dispatch<fork>");
                }
            }

            return p.get_future().get();
        }

        // launch::sync execute inline
        template <typename F, typename ...Ts>
        HPX_FORCEINLINE static
        typename hpx::util::detail::invoke_deferred_result<F, Ts...>::type
        call(launch::sync_policy, F && f, Ts&&... ts)
        {
            try {
                return hpx::util::invoke(
                    std::forward<F>(f), std::forward<Ts>(ts)...);
            }
            catch (std::bad_alloc const& ba) {
                throw ba;
            }
            catch (...) {
                throw exception_list(std::current_exception());
            }
        }
    };
}}

#endif
