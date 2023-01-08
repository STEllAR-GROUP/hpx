//  Copyright (c) 2007-2022 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file hpx/async_distributed/trigger_lco.hpp

#pragma once

#include <hpx/config.hpp>
#include <hpx/actions/actions_fwd.hpp>
#include <hpx/actions_base/action_priority.hpp>
#include <hpx/assert.hpp>
#include <hpx/async_distributed/continuation_fwd.hpp>
#include <hpx/async_distributed/detail/post_continue_fwd.hpp>
#include <hpx/async_distributed/detail/post_implementations_fwd.hpp>
#include <hpx/async_distributed/lcos_fwd.hpp>
#include <hpx/async_distributed/trigger_lco_fwd.hpp>
#include <hpx/components_base/component_type.hpp>
#include <hpx/naming_base/address.hpp>
#include <hpx/naming_base/id_type.hpp>
#include <hpx/type_support/unused.hpp>

#include <exception>
#include <type_traits>
#include <utility>

namespace hpx {

    /// \cond NOINTERNAL

    //////////////////////////////////////////////////////////////////////////
    // forward declare the required overload of post.
    template <typename Action, typename... Ts>
    bool post(hpx::id_type const& gid, Ts&&... vs);

    template <typename Component, typename Signature, typename Derived,
        typename... Ts>
    inline bool post_c_p(
        hpx::actions::basic_action<Component, Signature, Derived>,
        hpx::id_type const& contgid, hpx::id_type const& gid, Ts&&... vs);
    /// \endcond

    /// \cond NOINTERNAL
    namespace detail {
        template <typename T>
        struct make_rvalue_impl
        {
            typedef T&& type;

            template <typename U>
            HPX_FORCEINLINE static T&& call(U& u)
            {
                return HPX_MOVE(u);
            }
        };

        template <typename T>
        struct make_rvalue_impl<T const>
        {
            typedef T type;

            template <typename U>
            HPX_FORCEINLINE static T call(U const& u)
            {
                return u;
            }
        };

        template <typename T>
        struct make_rvalue_impl<T&>
        {
            typedef T type;

            HPX_FORCEINLINE static T call(T& u)
            {
                return u;
            }
        };

        template <typename T>
        struct make_rvalue_impl<T const&>
        {
            typedef T type;

            HPX_FORCEINLINE static T call(T const& u)
            {
                return u;
            }
        };

        template <typename T>
        HPX_FORCEINLINE typename detail::make_rvalue_impl<T>::type make_rvalue(
            typename std::remove_reference<T>::type& v)
        {
            return detail::make_rvalue_impl<T>::call(v);
        }

        template <typename T>
        HPX_FORCEINLINE typename detail::make_rvalue_impl<T>::type make_rvalue(
            typename std::remove_reference<T>::type&& v)
        {
            return detail::make_rvalue_impl<T>::call(v);
        }

        ///////////////////////////////////////////////////////////////////////
        template <typename Action, typename Result>
        void set_lco_value(hpx::id_type const& id, naming::address&& addr,
            Result&& t, bool move_credits)
        {
            if (move_credits &&
                id.get_management_type() !=
                    hpx::id_type::management_type::unmanaged)
            {
                hpx::id_type target(id.get_gid(),
                    hpx::id_type::management_type::managed_move_credit);
                id.make_unmanaged();

                detail::post_impl<Action>(target, HPX_MOVE(addr),
                    actions::action_priority<Action>(),
                    detail::make_rvalue<Result>(t));
            }
            else
            {
                detail::post_impl<Action>(id, HPX_MOVE(addr),
                    actions::action_priority<Action>(),
                    detail::make_rvalue<Result>(t));
            }
        }

        ///////////////////////////////////////////////////////////////////////
        template <typename LocalResult, typename RemoteResult, typename Action,
            typename Result>
        void set_lco_value(hpx::id_type const& id, naming::address&& addr,
            Result&& t, hpx::id_type const& cont, bool move_credits)
        {
            if (move_credits &&
                id.get_management_type() !=
                    hpx::id_type::management_type::unmanaged)
            {
                hpx::id_type target(id.get_gid(),
                    hpx::id_type::management_type::managed_move_credit);
                id.make_unmanaged();

                detail::post_impl<Action>(
                    actions::typed_continuation<LocalResult, RemoteResult>(
                        cont),
                    target, HPX_MOVE(addr), detail::make_rvalue<Result>(t));
            }
            else
            {
                detail::post_impl<Action>(
                    actions::typed_continuation<LocalResult, RemoteResult>(
                        cont),
                    id, HPX_MOVE(addr), detail::make_rvalue<Result>(t));
            }
        }
    }    // namespace detail
    /// \endcond

    /// \cond NOINTERNAL
    template <typename Result>
    void set_lco_value(hpx::id_type const& id, naming::address&& addr,
        Result&& t, bool move_credits)
    {
#if !defined(HPX_COMPUTE_DEVICE_CODE)
        typedef typename std::decay<Result>::type remote_result_type;
        typedef typename traits::promise_local_result<remote_result_type>::type
            local_result_type;

        if (components::get_base_type(addr.type_) ==
            components::component_base_lco_with_value_unmanaged)
        {
            typedef typename lcos::base_lco_with_value<local_result_type,
                remote_result_type,
                traits::detail::component_tag>::set_value_action
                set_value_action;

            detail::set_lco_value<set_value_action>(
                id, HPX_MOVE(addr), HPX_FORWARD(Result, t), move_credits);
        }
        else
        {
            // NOLINTNEXTLINE(bugprone-use-after-move)
            HPX_ASSERT(!addr ||
                components::get_base_type(addr.type_) ==
                    components::component_base_lco_with_value);

            typedef typename lcos::base_lco_with_value<local_result_type,
                remote_result_type,
                traits::detail::managed_component_tag>::set_value_action
                set_value_action;

            detail::set_lco_value<set_value_action>(
                id, HPX_MOVE(addr), HPX_FORWARD(Result, t), move_credits);
        }
#else
        HPX_UNUSED(id);
        HPX_UNUSED(addr);
        HPX_UNUSED(t);
        HPX_UNUSED(move_credits);
        HPX_ASSERT(false);
#endif
    }

    template <typename Result>
    void set_lco_value(hpx::id_type const& id, naming::address&& addr,
        Result&& t, hpx::id_type const& cont, bool move_credits)
    {
#if !defined(HPX_COMPUTE_DEVICE_CODE)
        typedef typename std::decay<Result>::type remote_result_type;
        typedef typename traits::promise_local_result<remote_result_type>::type
            local_result_type;

        if (components::get_base_type(addr.type_) ==
            components::component_base_lco_with_value_unmanaged)
        {
            typedef typename lcos::base_lco_with_value<local_result_type,
                remote_result_type,
                traits::detail::component_tag>::set_value_action
                set_value_action;

            detail::set_lco_value<local_result_type, remote_result_type,
                set_value_action>(
                id, HPX_MOVE(addr), HPX_FORWARD(Result, t), cont, move_credits);
        }
        else
        {
            HPX_ASSERT(!addr ||
                components::get_base_type(addr.type_) ==
                    components::component_base_lco_with_value);

            typedef typename lcos::base_lco_with_value<local_result_type,
                remote_result_type,
                traits::detail::managed_component_tag>::set_value_action
                set_value_action;

            detail::set_lco_value<local_result_type, remote_result_type,
                set_value_action>(
                id, HPX_MOVE(addr), HPX_FORWARD(Result, t), cont, move_credits);
        }
#else
        HPX_UNUSED(id);
        HPX_UNUSED(addr);
        HPX_UNUSED(t);
        HPX_UNUSED(cont);
        HPX_UNUSED(move_credits);
        HPX_ASSERT(false);
#endif
    }
    /// \endcond
}    // namespace hpx

#include <hpx/async_distributed/detail/post.hpp>
