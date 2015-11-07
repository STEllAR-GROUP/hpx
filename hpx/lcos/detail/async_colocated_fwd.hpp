//  Copyright (c) 2007-2015 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_LCOS_ASYNC_COLOCATED_FWD_FEB_01_2014_0107PM)
#define HPX_LCOS_ASYNC_COLOCATED_FWD_FEB_01_2014_0107PM

#include <hpx/config.hpp>
#include <hpx/runtime/actions/basic_action_fwd.hpp>
#include <hpx/runtime/naming/name.hpp>
#include <hpx/traits/is_continuation.hpp>
#include <hpx/traits/promise_local_result.hpp>
#include <hpx/lcos/async_fwd.hpp>
#include <hpx/util/move.hpp>

namespace hpx { namespace detail
{
    ///////////////////////////////////////////////////////////////////////////
    template <typename Action, typename ...Ts>
    lcos::future<
        typename traits::promise_local_result<
            typename hpx::actions::extract_action<Action>::remote_result_type
        >::type>
    async_colocated(naming::id_type const& gid, Ts&&... vs);

    template <
        typename Component, typename Signature, typename Derived,
        typename ...Ts>
    lcos::future<
        typename traits::promise_local_result<
            typename hpx::actions::extract_action<Derived>::remote_result_type
        >::type>
    async_colocated(
        hpx::actions::basic_action<Component, Signature, Derived> /*act*/
      , naming::id_type const& gid, Ts&&... vs);

    ///////////////////////////////////////////////////////////////////////////
    // MSVC complains about ambiguities if it sees this forward declaration
#if !defined(BOOST_MSVC)
    template <typename Action, typename Continuation, typename ...Ts>
    typename std::enable_if<
        traits::is_continuation<Continuation>::value,
        lcos::future<
            typename traits::promise_local_result<
                typename hpx::actions::extract_action<Action>::remote_result_type
            >::type
        >
    >::type
    async_colocated(Continuation && cont,
        naming::id_type const& gid, Ts&&... vs);

    template <
        typename Continuation,
        typename Component, typename Signature, typename Derived,
        typename ...Ts>
    typename std::enable_if<
        traits::is_continuation<Continuation>::value,
        lcos::future<
            typename traits::promise_local_result<
                typename hpx::actions::extract_action<Derived>::remote_result_type
            >::type
        >
    >::type
    async_colocated(
        Continuation && cont
      , hpx::actions::basic_action<Component, Signature, Derived> /*act*/
      , naming::id_type const& gid, Ts&&... vs);
#endif
}}

#if defined(HPX_HAVE_COLOCATED_BACKWARDS_COMPATIBILITY)
namespace hpx
{
    using hpx::detail::async_colocated;
}
#endif

#endif
