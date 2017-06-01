//  Copyright (c) 2007-2013 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

///////////////////////////////////////////////////////////////////////////////

#if !defined(HPX_LCOS_ASYNC_CONTINUE_FWD_JAN_25_2013_0828AM)
#define HPX_LCOS_ASYNC_CONTINUE_FWD_JAN_25_2013_0828AM

#include <hpx/config.hpp>
#include <hpx/lcos/future.hpp>
#include <hpx/traits/action_remote_result.hpp>
#include <hpx/traits/extract_action.hpp>
#include <hpx/traits/is_distribution_policy.hpp>
#include <hpx/traits/promise_local_result.hpp>
#include <hpx/util/decay.hpp>
#include <hpx/util/result_of.hpp>

#ifndef HPX_MSVC
#include <type_traits>
#endif

namespace hpx
{
    ///////////////////////////////////////////////////////////////////////////
    namespace detail
    {
        template <typename Action, typename Cont>
        struct result_of_async_continue
            : traits::action_remote_result<
                typename util::invoke_result<typename util::decay<Cont>::type,
                    naming::id_type,
                    typename hpx::traits::extract_action<
                        Action
                    >::remote_result_type
                >::type
            >
        {};

        template <
            typename Action, typename RemoteResult, typename Cont,
            typename Target, typename ...Ts>
        lcos::future<
            typename traits::promise_local_result<
                typename result_of_async_continue<Action, Cont>::type
            >::type
        >
        async_continue_r(Cont&& cont, Target const& target, Ts&&... vs);
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename Action, typename Cont, typename ...Ts>
    lcos::future<
        typename traits::promise_local_result<
            typename detail::result_of_async_continue<Action, Cont>::type
        >::type
    >
    async_continue(Cont&& cont, naming::id_type const& gid, Ts&&... vs);

    template <
        typename Component, typename Signature, typename Derived,
        typename Cont, typename ...Ts>
    lcos::future<
        typename traits::promise_local_result<
            typename detail::result_of_async_continue<Derived, Cont>::type
        >::type
    >
    async_continue(
        hpx::actions::basic_action<Component, Signature, Derived> /*act*/
      , Cont&& cont, naming::id_type const& gid, Ts&&... vs);

    ///////////////////////////////////////////////////////////////////////////
    // MSVC complains about ambiguities if it sees this forward declaration
#ifndef HPX_MSVC
    template <typename Action, typename Cont, typename DistPolicy,
        typename ...Ts>
    typename std::enable_if<
        traits::is_distribution_policy<DistPolicy>::value,
        lcos::future<
            typename traits::promise_local_result<
                typename detail::result_of_async_continue<Action, Cont>::type
            >::type>
    >::type
    async_continue(Cont&& cont, DistPolicy const& policy, Ts&&... vs);

    template <
        typename Component, typename Signature, typename Derived,
        typename Cont, typename DistPolicy, typename ...Ts>
    typename std::enable_if<
        traits::is_distribution_policy<DistPolicy>::value,
        lcos::future<
            typename traits::promise_local_result<
                typename detail::result_of_async_continue<Derived, Cont>::type
            >::type>
    >::type
    async_continue(
        hpx::actions::basic_action<Component, Signature, Derived> /*act*/
      , Cont&& cont, DistPolicy const& policy, Ts&&... vs);
#endif
}

#endif
