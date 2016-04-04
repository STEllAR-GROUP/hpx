//  Copyright (c) 2007-2015 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_LCOS_ASYNC_COLOCATED_CALLBACK_FWD_MAR_30_2015_1145PM)
#define HPX_LCOS_ASYNC_COLOCATED_CALLBACK_FWD_MAR_30_2015_1145PM

#include <hpx/traits/extract_action.hpp>
#include <hpx/lcos/detail/async_colocated_fwd.hpp>

namespace hpx { namespace detail
{
    ///////////////////////////////////////////////////////////////////////////
    template <typename Action, typename Callback, typename ...Ts>
    lcos::future<
        typename traits::promise_local_result<
            typename hpx::actions::extract_action<Action>::remote_result_type
        >::type>
    async_colocated_cb(naming::id_type const& gid, Callback&& cb, Ts&&... vs);

    template <
        typename Component, typename Signature, typename Derived,
        typename Callback, typename ...Ts>
    lcos::future<
        typename traits::promise_local_result<
            typename hpx::actions::extract_action<Derived>::remote_result_type
        >::type>
    async_colocated_cb(
        hpx::actions::basic_action<Component, Signature, Derived> /*act*/
      , naming::id_type const& gid, Callback&& cb, Ts&&... vs);

    ///////////////////////////////////////////////////////////////////////////
    template <typename Action, typename Continuation, typename Callback, typename ...Ts>
    lcos::future<
        typename traits::promise_local_result<
            typename hpx::actions::extract_action<Action>::remote_result_type
        >::type>
    async_colocated_cb(Continuation && cont,
        naming::id_type const& gid, Callback&& cb, Ts&&... vs);

    template <
        typename Continuation,
        typename Component, typename Signature, typename Derived,
        typename Callback, typename ...Ts>
    lcos::future<
        typename traits::promise_local_result<
            typename hpx::actions::extract_action<Derived>::remote_result_type
        >::type>
    async_colocated_cb(
        Continuation && cont
      , hpx::actions::basic_action<Component, Signature, Derived> /*act*/
      , naming::id_type const& gid, Callback&& cb, Ts&&... vs);
}}

#if defined(HPX_HAVE_COLOCATED_BACKWARDS_COMPATIBILITY)
namespace hpx
{
    using hpx::detail::async_colocated_cb;
}
#endif

#endif
