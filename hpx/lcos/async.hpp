//  Copyright (c) 2007-2015 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_LCOS_ASYNC_SEP_28_2011_0840AM)
#define HPX_LCOS_ASYNC_SEP_28_2011_0840AM

#include <hpx/hpx_fwd.hpp>
#include <hpx/traits/is_distribution_policy.hpp>
#include <hpx/runtime/launch_policy.hpp>
#include <hpx/lcos/detail/async_implementations.hpp>
#include <hpx/lcos/future.hpp>
#include <hpx/lcos/async_fwd.hpp>

    ///////////////////////////////////////////////////////////////////////////
namespace hpx { namespace detail
{
    // BOOST_SCOPED_ENUM(launch)
    template <typename Action, typename Policy>
    struct async_action_dispatch<Action, Policy,
        typename boost::enable_if_c<
            traits::is_launch_policy<Policy>::value
        >::type>
    {
        template <typename ...Ts>
        BOOST_FORCEINLINE static lcos::future<
            typename traits::promise_local_result<
                typename hpx::actions::extract_action<
                    Action
                >::remote_result_type
            >::type>
        call(BOOST_SCOPED_ENUM(launch) launch_policy,
            naming::id_type const& id, Ts&&... ts)
        {
            return hpx::detail::async_impl<Action>(launch_policy, id,
                std::forward<Ts>(ts)...);
        }

        template <typename DistPolicy, typename ...Ts>
        BOOST_FORCEINLINE static
        typename boost::enable_if_c<
            traits::is_distribution_policy<DistPolicy>::value,
            lcos::future<
                typename traits::promise_local_result<
                    typename hpx::actions::extract_action<
                        Action
                    >::remote_result_type
                >::type
            >
        >::type
        call(BOOST_SCOPED_ENUM(launch) launch_policy,
            DistPolicy const& policy, Ts&&... ts)
        {
            return policy.template async<Action>(launch_policy,
                std::forward<Ts>(ts)...);
        }
    };

    // naming::id_type
    template <typename Action>
    struct async_action_dispatch<Action, naming::id_type>
    {
        template <typename ...Ts>
        BOOST_FORCEINLINE static
        lcos::future<
            typename traits::promise_local_result<
                typename hpx::actions::extract_action<
                    Action
                >::remote_result_type
            >::type>
        call(naming::id_type const& id, Ts&&... ts)
        {
            return async_action_dispatch<
                    Action, BOOST_SCOPED_ENUM(launch)
                >::call(launch::all, id, std::forward<Ts>(ts)...);
        }
    };

    // distribution policy
    template <typename Action, typename Policy>
    struct async_action_dispatch<Action, Policy,
        typename boost::enable_if_c<
            traits::is_distribution_policy<Policy>::value
        >::type>
    {
        template <typename DistPolicy, typename ...Ts>
        BOOST_FORCEINLINE static
        lcos::future<
            typename traits::promise_local_result<
                typename hpx::actions::extract_action<
                    Action
                >::remote_result_type
            >::type>
        call(DistPolicy const& policy, Ts&&... ts)
        {
            return async_action_dispatch<
                    Action, BOOST_SCOPED_ENUM(launch)
                >::call(launch::all, policy, std::forward<Ts>(ts)...);
        }
    };
}}

namespace hpx
{
    template <typename Action, typename F, typename ...Ts>
    BOOST_FORCEINLINE
    auto async(F&& f, Ts&&... ts)
    ->  decltype(detail::async_action_dispatch<
                    Action, typename util::decay<F>::type
            >::call(std::forward<F>(f), std::forward<Ts>(ts)...))
    {
        return detail::async_action_dispatch<
                Action, typename util::decay<F>::type
            >::call(std::forward<F>(f), std::forward<Ts>(ts)...);
    }
}

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace detail
{
    ///////////////////////////////////////////////////////////////////////////
    // any action
    template <typename Action>
    struct async_dispatch<Action,
        typename boost::enable_if_c<
            traits::is_action<Action>::value
        >::type>
    {
        template <
            typename Component, typename Signature, typename Derived,
            typename ...Ts>
        BOOST_FORCEINLINE static lcos::future<
            typename traits::promise_local_result<
                typename hpx::actions::extract_action<
                    Derived
                >::remote_result_type
            >::type>
        call(hpx::actions::basic_action<Component, Signature, Derived> const&,
            naming::id_type const& id, Ts&&... vs)
        {
            return async<Derived>(launch::all, id, std::forward<Ts>(vs)...);
        }

        template <
            typename Component, typename Signature, typename Derived,
            typename DistPolicy, typename ...Ts>
        BOOST_FORCEINLINE static lcos::future<
            typename traits::promise_local_result<
                typename hpx::actions::extract_action<
                    Derived
                >::remote_result_type
            >::type>
        call(hpx::actions::basic_action<Component, Signature, Derived> const&,
            DistPolicy const& policy, Ts&&... vs)
        {
            return async<Derived>(policy, std::forward<Ts>(vs)...);
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename Policy>
    struct async_dispatch<Policy,
        typename boost::enable_if_c<
            traits::is_launch_policy<Policy>::value
         && traits::is_action<Policy>::value
        >::type>
    {
        template <typename Action, typename ...Ts>
        BOOST_FORCEINLINE static
        lcos::future<
            typename traits::promise_local_result<
                typename hpx::actions::extract_action<
                    Action
                >::remote_result_type
            >::type>
        call(BOOST_SCOPED_ENUM(launch) launch_policy,
            Action const&, naming::id_type const& id, Ts&&... ts)
        {
            return async<Action>(launch_policy, id, std::forward<Ts>(ts)...);
        }

        template <typename Action, typename DistPolicy, typename ...Ts>
        BOOST_FORCEINLINE static
        typename boost::enable_if_c<
            traits::is_distribution_policy<DistPolicy>::value,
            lcos::future<
                typename traits::promise_local_result<
                    typename hpx::actions::extract_action<
                        Action
                    >::remote_result_type
                >::type>
        >::type
        call(BOOST_SCOPED_ENUM(launch) launch_policy,
            Action const&, DistPolicy const& policy, Ts&&... ts)
        {
            return async<Action>(launch_policy, policy, std::forward<Ts>(ts)...);
        }
    };
}}

#endif
