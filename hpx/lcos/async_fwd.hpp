//  Copyright (c) 2007-2013 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

///////////////////////////////////////////////////////////////////////////////

#if !defined(HPX_LCOS_ASYNC_FWD_SEP_28_2011_0840AM)
#define HPX_LCOS_ASYNC_FWD_SEP_28_2011_0840AM

#include <hpx/hpx_fwd.hpp>
#include <hpx/traits.hpp>
#include <hpx/util/always_void.hpp>
#include <hpx/util/move.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace actions
{
    // This template meta function can be used to extract the action type, no
    // matter whether it got specified directly or by passing the
    // corresponding make_action<> specialization.
    template <typename Action, typename Enable = void>
    struct extract_action
    {
        typedef typename Action::derived_type type;
        typedef typename type::result_type result_type;
        typedef typename type::remote_result_type remote_result_type;
    };

    template <typename Action>
    struct extract_action<Action
      , typename util::always_void<typename Action::type>::type>
      : extract_action<typename Action::type>
    {};
}}

///////////////////////////////////////////////////////////////////////////////
namespace hpx
{
    ///////////////////////////////////////////////////////////////////////////
    template <typename Action, typename ...Ts>
    lcos::future<
        typename traits::promise_local_result<
            typename hpx::actions::extract_action<Action>::remote_result_type
        >::type>
    async(BOOST_SCOPED_ENUM(launch) policy, naming::id_type const& gid,
        Ts&&... vs);

    template <typename Action, typename ...Ts>
    lcos::future<
        typename traits::promise_local_result<
            typename hpx::actions::extract_action<Action>::remote_result_type
        >::type>
    async(naming::id_type const& gid, Ts&&... vs);

    template <
        typename Component, typename Signature, typename Derived,
        typename ...Ts>
    lcos::future<
        typename traits::promise_local_result<
            typename hpx::actions::extract_action<Derived>::remote_result_type
        >::type>
    async(
        hpx::actions::basic_action<Component, Signature, Derived> const& /*act*/,
        naming::id_type const& gid, Ts&&... vs);

    template <
        typename Component, typename Signature, typename Derived,
        typename ...Ts>
    lcos::future<
        typename traits::promise_local_result<
            typename hpx::actions::extract_action<Derived>::remote_result_type
        >::type>
    async(BOOST_SCOPED_ENUM(launch) policy,
        hpx::actions::basic_action<Component, Signature, Derived> const& /*act*/,
        naming::id_type const& gid, Ts&&... vs);
}

#endif
