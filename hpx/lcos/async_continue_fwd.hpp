//  Copyright (c) 2007-2013 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

///////////////////////////////////////////////////////////////////////////////

#if !defined(HPX_LCOS_ASYNC_CONTINUE_FWD_JAN_25_2013_0828AM)
#define HPX_LCOS_ASYNC_CONTINUE_FWD_JAN_25_2013_0828AM

#include <hpx/config.hpp>
#include <hpx/traits/extract_action.hpp>
#include <hpx/util/decay.hpp>
#include <hpx/util/result_of.hpp>
#include <hpx/lcos/future.hpp>

#ifndef HPX_MSVC
#include <boost/utility/enable_if.hpp>
#endif

namespace hpx
{
    ///////////////////////////////////////////////////////////////////////////
    namespace actions { namespace detail
    {
        template <typename Result>
        struct remote_action_result;
    }}

    ///////////////////////////////////////////////////////////////////////////
    namespace detail
    {
        template <typename Action, typename Cont>
        struct result_of_async_continue
            : actions::detail::remote_action_result<
                typename util::result_of<typename util::decay<Cont>::type(
                    naming::id_type,
                    typename hpx::actions::extract_action<
                        Action
                    >::remote_result_type
                )>::type
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
    typename boost::enable_if_c<
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
    typename boost::enable_if_c<
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
