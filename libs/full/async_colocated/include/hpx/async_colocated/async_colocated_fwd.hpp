//  Copyright (c) 2007-2015 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/actions_base/basic_action_fwd.hpp>
#include <hpx/actions_base/traits/extract_action.hpp>
#include <hpx/async_local/async_fwd.hpp>
#include <hpx/futures/future.hpp>
#include <hpx/futures/traits/promise_local_result.hpp>
#include <hpx/naming_base/id_type.hpp>
#include <hpx/traits/is_continuation.hpp>

#include <type_traits>

namespace hpx { namespace detail {
    ///////////////////////////////////////////////////////////////////////////
    template <typename Action, typename... Ts>
    lcos::future<typename traits::promise_local_result<
        typename hpx::traits::extract_action<Action>::remote_result_type>::type>
    async_colocated(naming::id_type const& id, Ts&&... vs);

    template <typename Component, typename Signature, typename Derived,
        typename... Ts>
    lcos::future<typename traits::promise_local_result<typename hpx::traits::
            extract_action<Derived>::remote_result_type>::type>
    async_colocated(
        hpx::actions::basic_action<Component, Signature, Derived> /*act*/
        ,
        naming::id_type const& id, Ts&&... vs);

    ///////////////////////////////////////////////////////////////////////////
    // MSVC complains about ambiguities if it sees this forward declaration
#if !defined(HPX_MSVC)
    template <typename Action, typename Continuation, typename... Ts>
    typename std::enable_if<traits::is_continuation<Continuation>::value,
        lcos::future<typename traits::promise_local_result<typename hpx::
                traits::extract_action<Action>::remote_result_type>::type>>::
        type
        async_colocated(
            Continuation&& cont, naming::id_type const& id, Ts&&... vs);

    template <typename Continuation, typename Component, typename Signature,
        typename Derived, typename... Ts>
    typename std::enable_if<traits::is_continuation<Continuation>::value,
        lcos::future<typename traits::promise_local_result<typename hpx::
                traits::extract_action<Derived>::remote_result_type>::type>>::
        type
        async_colocated(Continuation&& cont,
            hpx::actions::basic_action<Component, Signature, Derived> /*act*/
            ,
            naming::id_type const& id, Ts&&... vs);
#endif
}}    // namespace hpx::detail

#if defined(HPX_HAVE_COLOCATED_BACKWARDS_COMPATIBILITY)
namespace hpx {
    using hpx::detail::async_colocated;
}
#endif
