//  Copyright (c) 2007-2023 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Lelbach
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/actions/post_helper.hpp>
#include <hpx/actions_base/action_priority.hpp>
#include <hpx/actions_base/actions_base_support.hpp>
#include <hpx/actions_base/traits/extract_action.hpp>
#include <hpx/actions_base/traits/is_continuation.hpp>
#include <hpx/actions_base/traits/is_distribution_policy.hpp>
#include <hpx/actions_base/traits/is_valid_action.hpp>
#include <hpx/assert.hpp>
#include <hpx/async_distributed/continuation.hpp>
#include <hpx/async_distributed/detail/post_implementations_fwd.hpp>
#include <hpx/async_distributed/put_parcel_fwd.hpp>
#include <hpx/async_local/post.hpp>
#include <hpx/components/client_base.hpp>
#include <hpx/components_base/agas_interface.hpp>
#include <hpx/components_base/traits/component_type_is_compatible.hpp>
#include <hpx/functional/traits/is_action.hpp>
#include <hpx/modules/threading_base.hpp>
#include <hpx/naming_base/address.hpp>
#include <hpx/naming_base/id_type.hpp>
#include <hpx/parcelset/detail/parcel_await.hpp>
#include <hpx/parcelset/parcelset_fwd.hpp>
#include <hpx/parcelset_base/parcel_interface.hpp>
#include <hpx/runtime_local/get_locality_id.hpp>

#include <type_traits>
#include <utility>

namespace hpx {

    ///////////////////////////////////////////////////////////////////////////
    // Invoked by a running HPX-thread to post() an action to any resource
    namespace detail {
#if defined(HPX_HAVE_NETWORKING)
        ///////////////////////////////////////////////////////////////////////
        template <typename Action>
        inline naming::address&& complement_addr(naming::address& addr)
        {
            if (components::component_invalid == addr.type_)
            {
                addr.type_ = components::get_component_type<
                    typename Action::component_type>();
            }
            return HPX_MOVE(addr);
        }

        template <typename Action, typename... Ts>
        inline bool put_parcel(hpx::id_type const& id, naming::address&& addr,
            threads::thread_priority priority, Ts&&... vs)
        {
            typedef
                typename hpx::traits::extract_action<Action>::type action_type;
            action_type act;

            parcelset::put_parcel(id, complement_addr<action_type>(addr), act,
                priority, HPX_FORWARD(Ts, vs)...);

            return false;    // destinations are remote
        }

        template <typename Action, typename Continuation, typename... Ts>
        inline bool put_parcel_cont(hpx::id_type const& id,
            naming::address&& addr, threads::thread_priority priority,
            Continuation&& cont, Ts&&... vs)
        {
            typedef
                typename hpx::traits::extract_action<Action>::type action_type;
            action_type act;

            parcelset::put_parcel(id, complement_addr<action_type>(addr),
                HPX_FORWARD(Continuation, cont), act, priority,
                HPX_FORWARD(Ts, vs)...);

            return false;    // destinations are remote
        }

        template <typename Action, typename... Ts>
        inline bool put_parcel_cb(hpx::id_type const& id,
            naming::address&& addr, threads::thread_priority priority,
            parcelset::write_handler_type const& cb, Ts&&... vs)
        {
            typedef
                typename hpx::traits::extract_action<Action>::type action_type;
            action_type act;

            parcelset::put_parcel_cb(cb, id, complement_addr<action_type>(addr),
                act, priority, HPX_FORWARD(Ts, vs)...);

            return false;    // destinations are remote
        }

        template <typename Action, typename... Ts>
        inline bool put_parcel_cb(hpx::id_type const& id,
            naming::address&& addr, threads::thread_priority priority,
            parcelset::write_handler_type&& cb, Ts&&... vs)
        {
            typedef
                typename hpx::traits::extract_action<Action>::type action_type;
            action_type act;

            parcelset::put_parcel_cb(HPX_MOVE(cb), id,
                complement_addr<action_type>(addr), act, priority,
                HPX_FORWARD(Ts, vs)...);

            return false;    // destinations are remote
        }

        template <typename Action, typename Continuation, typename... Ts>
        inline bool put_parcel_cont_cb(hpx::id_type const& id,
            naming::address&& addr, threads::thread_priority priority,
            Continuation&& cont, parcelset::write_handler_type const& cb,
            Ts&&... vs)
        {
            typedef
                typename hpx::traits::extract_action<Action>::type action_type;
            action_type act;

            parcelset::put_parcel_cb(cb, id, complement_addr<action_type>(addr),
                HPX_FORWARD(Continuation, cont), act, priority,
                HPX_FORWARD(Ts, vs)...);

            return false;    // destinations are remote
        }

        template <typename Action, typename Continuation, typename... Ts>
        inline bool put_parcel_cont_cb(hpx::id_type const& id,
            naming::address&& addr, threads::thread_priority priority,
            Continuation&& cont, parcelset::write_handler_type&& cb, Ts&&... vs)
        {
            typedef
                typename hpx::traits::extract_action<Action>::type action_type;
            action_type act;

            parcelset::put_parcel_cb(HPX_MOVE(cb), id,
                complement_addr<action_type>(addr),
                HPX_FORWARD(Continuation, cont), act, priority,
                HPX_FORWARD(Ts, vs)...);

            return false;    // destinations are remote
        }

        // We know it is remote.
        template <typename Action, typename... Ts>
        inline bool post_r_p(naming::address&& addr, hpx::id_type const& id,
            threads::thread_priority priority, Ts&&... vs)
        {
            // If remote, create a new parcel to be sent to the destination
            // Create a new parcel with the gid, action, and arguments
            return detail::put_parcel<Action>(
                id, HPX_MOVE(addr), priority, HPX_FORWARD(Ts, vs)...);
        }

        template <typename Action, typename... Ts>
        inline bool post_r(
            naming::address&& addr, hpx::id_type const& gid, Ts&&... vs)
        {
            return post_r_p<Action>(HPX_MOVE(addr), gid,
                actions::action_priority<Action>(), HPX_FORWARD(Ts, vs)...);
        }
#endif

        // We know it is local and has to be directly executed.
        template <typename Action, typename... Ts>
        inline bool post_l_p(hpx::id_type const& target, naming::address&& addr,
            threads::thread_priority priority, Ts&&... vs)
        {
            typedef
                typename hpx::traits::extract_action<Action>::type action_type;

            HPX_ASSERT(traits::component_type_is_compatible<
                typename action_type::component_type>::call(addr));

            threads::thread_init_data data;
#if defined(HPX_HAVE_THREAD_DESCRIPTION)
            data.description = actions::detail::get_action_name<Action>();
#endif
#if defined(HPX_HAVE_THREAD_PARENT_REFERENCE)
            data.parent_id = threads::get_self_id();
            data.parent_locality_id = get_locality_id();
#endif
#if defined(HPX_HAVE_APEX)
            data.timer_data = hpx::util::external_timer::new_task(
                data.description, data.parent_locality_id, data.parent_id);
#endif
            post_helper<action_type>::call(HPX_MOVE(data), target,
                addr.address_, addr.type_, priority, HPX_FORWARD(Ts, vs)...);
            return true;    // no parcel has been sent (dest is local)
        }

        // same as above, but taking all arguments by value
        template <typename Action, typename... Ts>
        inline bool post_l_p_val(hpx::id_type const& target,
            naming::address&& addr, threads::thread_priority priority, Ts... vs)
        {
            typedef
                typename hpx::traits::extract_action<Action>::type action_type;

            HPX_ASSERT(traits::component_type_is_compatible<
                typename action_type::component_type>::call(addr));

            threads::thread_init_data data;
#if defined(HPX_HAVE_THREAD_DESCRIPTION)
            data.description = actions::detail::get_action_name<Action>();
#endif
#if defined(HPX_HAVE_THREAD_PARENT_REFERENCE)
            data.parent_id = threads::get_self_id();
            data.parent_locality_id = get_locality_id();
#endif
#if defined(HPX_HAVE_APEX)
            data.timer_data = hpx::util::external_timer::new_task(
                data.description, data.parent_locality_id, data.parent_id);
#endif
            post_helper<action_type>::call(HPX_MOVE(data), target,
                addr.address_, addr.type_, priority, HPX_MOVE(vs)...);
            return true;    // no parcel has been sent (dest is local)
        }

        template <typename Action, typename... Ts>
        inline bool post_l(
            hpx::id_type const& target, naming::address&& addr, Ts&&... vs)
        {
            return post_l_p<Action>(target, HPX_MOVE(addr),
                actions::action_priority<Action>(), HPX_FORWARD(Ts, vs)...);
        }
    }    // namespace detail

    ///////////////////////////////////////////////////////////////////////////
    template <typename Action, typename... Ts>
    inline bool post_p(
        hpx::id_type const& id, threads::thread_priority priority, Ts&&... vs)
    {
        return hpx::detail::post_impl<Action>(
            id, priority, HPX_FORWARD(Ts, vs)...);
    }

    template <typename Action, typename Client, typename Stub, typename Data,
        typename... Ts>
    inline bool post_p(components::client_base<Client, Stub, Data> const& c,
        threads::thread_priority priority, Ts&&... vs)
    {
        // make sure the action is compatible with the component type
        typedef typename components::client_base<Client, Stub,
            Data>::server_component_type component_type;

        static_assert(traits::is_valid_action_v<Action, component_type>,
            "The action to invoke is not supported by the target");

        return hpx::detail::post_impl<Action>(
            c.get_id(), priority, HPX_FORWARD(Ts, vs)...);
    }

    template <typename Action, typename DistPolicy, typename... Ts>
    std::enable_if_t<traits::is_distribution_policy_v<DistPolicy>, bool> post_p(
        DistPolicy const& policy, threads::thread_priority priority, Ts&&... vs)
    {
        return policy.template apply<Action>(priority, HPX_FORWARD(Ts, vs)...);
    }

    namespace detail {

        template <typename Action>
        struct post_dispatch<Action,
            std::enable_if_t<traits::is_action<Action>::value>>
        {
            template <typename Component, typename Signature, typename Derived,
                typename... Ts>
            HPX_FORCEINLINE static bool call(
                hpx::actions::basic_action<Component, Signature, Derived>,
                hpx::id_type const& id, Ts&&... ts)
            {
                return hpx::post_p<Derived>(id,
                    actions::action_priority<Derived>(),
                    HPX_FORWARD(Ts, ts)...);
            }

            template <typename Component, typename Signature, typename Derived,
                typename Client, typename Stub, typename Data, typename... Ts>
            HPX_FORCEINLINE static bool call(
                hpx::actions::basic_action<Component, Signature, Derived>,
                components::client_base<Client, Stub, Data> const& c,
                Ts&&... ts)
            {
                // make sure the action is compatible with the component type
                typedef typename components::client_base<Client, Stub,
                    Data>::server_component_type component_type;

                static_assert(
                    traits::is_valid_action_v<Derived, component_type>,
                    "The action to invoke is not supported by the target");

                return hpx::post_p<Derived>(c.get_id(),
                    actions::action_priority<Derived>(),
                    HPX_FORWARD(Ts, ts)...);
            }

            template <typename Component, typename Signature, typename Derived,
                typename DistPolicy, typename... Ts>
            HPX_FORCEINLINE static std::enable_if_t<
                traits::is_distribution_policy_v<DistPolicy>, bool>
            call(hpx::actions::basic_action<Component, Signature, Derived>,
                DistPolicy const& policy, Ts&&... ts)
            {
                return hpx::post_p<Derived>(policy,
                    actions::action_priority<Derived>(),
                    HPX_FORWARD(Ts, ts)...);
            }
        };
    }    // namespace detail

    template <typename Action, typename... Ts>
    inline bool post(hpx::id_type const& id, Ts&&... vs)
    {
        return hpx::post_p<Action>(
            id, actions::action_priority<Action>(), HPX_FORWARD(Ts, vs)...);
    }

    template <typename Action, typename Client, typename Stub, typename Data,
        typename... Ts>
    inline bool post(
        components::client_base<Client, Stub, Data> const& c, Ts&&... vs)
    {
        // make sure the action is compatible with the component type
        typedef typename components::client_base<Client, Stub,
            Data>::server_component_type component_type;

        static_assert(traits::is_valid_action_v<Action, component_type>,
            "The action to invoke is not supported by the target");

        return hpx::post_p<Action>(c.get_id(),
            actions::action_priority<Action>(), HPX_FORWARD(Ts, vs)...);
    }

    template <typename Action, typename DistPolicy, typename... Ts>
    std::enable_if_t<traits::is_distribution_policy_v<DistPolicy>, bool> post(
        DistPolicy const& policy, Ts&&... vs)
    {
        return hpx::post_p<Action>(
            policy, actions::action_priority<Action>(), HPX_FORWARD(Ts, vs)...);
    }

    ///////////////////////////////////////////////////////////////////////////
    namespace detail {
#if defined(HPX_HAVE_NETWORKING)
        template <typename Action, typename Continuation, typename... Ts>
        inline bool post_r_p(naming::address&& addr, Continuation&& c,
            hpx::id_type const& id, threads::thread_priority priority,
            Ts&&... vs)
        {
            // If remote, create a new parcel to be sent to the destination
            // Create a new parcel with the gid, action, and arguments
            return detail::put_parcel_cont<Action>(id, HPX_MOVE(addr), priority,
                HPX_FORWARD(Continuation, c), HPX_FORWARD(Ts, vs)...);
        }

        template <typename Action, typename Continuation, typename... Ts>
        inline std::enable_if_t<traits::is_continuation<Continuation>::value,
            bool>
        post_r(naming::address&& addr, Continuation&& c,
            hpx::id_type const& gid, Ts&&... vs)
        {
            return post_r_p<Action>(HPX_MOVE(addr),
                HPX_FORWARD(Continuation, c), gid,
                actions::action_priority<Action>(), HPX_FORWARD(Ts, vs)...);
        }

        template <typename Action>
        inline bool post_r_sync_p(naming::address&& addr,
            hpx::id_type const& id, threads::thread_priority priority)
        {
            typedef
                typename hpx::traits::extract_action<Action>::type action_type_;

            // If remote, create a new parcel to be sent to the destination
            // Create a new parcel with the gid, action, and arguments
            // Send the parcel through the parcel handler
            HPX_ASSERT(id.get_management_type() ==
                hpx::id_type::management_type::unmanaged);
            naming::gid_type gid = id.get_gid();
            parcelset::parcel p = parcelset::detail::create_parcel::call(
                HPX_MOVE(gid), complement_addr<action_type_>(addr),
                action_type_(), priority);

            parcelset::detail::parcel_await_apply(HPX_MOVE(p),
                parcelset::write_handler_type(), 0,
                [](parcelset::parcel&& p, parcelset::write_handler_type&&) {
                    hpx::parcelset::sync_put_parcel(HPX_MOVE(p));
                });

            return false;    // destination is remote
        }

        template <typename Action>
        inline bool post_r_sync(naming::address&& addr, hpx::id_type const& gid)
        {
            return post_r_sync_p<Action>(
                HPX_MOVE(addr), gid, actions::action_priority<Action>());
        }
#endif

        // We know it is local and has to be directly executed.
        template <typename Action, typename Continuation, typename... Ts>
        inline bool post_l_p(Continuation&& cont, hpx::id_type const& target,
            naming::address&& addr, threads::thread_priority priority,
            Ts&&... vs)
        {
            typedef
                typename hpx::traits::extract_action<Action>::type action_type;

            HPX_ASSERT(traits::component_type_is_compatible<
                typename action_type::component_type>::call(addr));

            threads::thread_init_data data;
#if defined(HPX_HAVE_THREAD_DESCRIPTION)
            data.description = actions::detail::get_action_name<Action>();
#endif
#if defined(HPX_HAVE_THREAD_PARENT_REFERENCE)
            data.parent_id = threads::get_self_id();
            data.parent_locality_id = get_locality_id();
#endif
#if defined(HPX_HAVE_APEX)
            data.timer_data = hpx::util::external_timer::new_task(
                data.description, data.parent_locality_id, data.parent_id);
#endif
            post_helper<action_type>::call(HPX_MOVE(data),
                HPX_FORWARD(Continuation, cont), target, addr.address_,
                addr.type_, priority, HPX_FORWARD(Ts, vs)...);
            return true;    // no parcel has been sent (dest is local)
        }

        template <typename Action, typename Continuation, typename... Ts>
        inline std::enable_if_t<traits::is_continuation<Continuation>::value,
            bool>
        post_l(Continuation&& c, hpx::id_type const& target,
            naming::address& addr, Ts&&... vs)
        {
            return post_l_p<Action>(HPX_FORWARD(Continuation, c), target,
                HPX_MOVE(addr), actions::action_priority<Action>(),
                HPX_FORWARD(Ts, vs)...);
        }
    }    // namespace detail

    ///////////////////////////////////////////////////////////////////////////
    template <typename Action, typename Continuation, typename... Ts>
    inline std::enable_if_t<traits::is_continuation<Continuation>::value, bool>
    post_p(Continuation&& c, hpx::id_type const& gid,
        threads::thread_priority priority, Ts&&... vs)
    {
        return hpx::detail::post_impl<Action>(HPX_FORWARD(Continuation, c), gid,
            priority, HPX_FORWARD(Ts, vs)...);
    }

    template <typename Action, typename Continuation, typename Client,
        typename Stub, typename Data, typename... Ts>
    inline std::enable_if_t<traits::is_continuation<Continuation>::value, bool>
    post_p(Continuation&& cont,
        components::client_base<Client, Stub, Data> const& c,
        threads::thread_priority priority, Ts&&... vs)
    {
        // make sure the action is compatible with the component type
        typedef typename components::client_base<Client, Stub,
            Data>::server_component_type component_type;

        static_assert(traits::is_valid_action_v<Action, component_type>,
            "The action to invoke is not supported by the target");

        return hpx::detail::post_impl<Action>(HPX_FORWARD(Continuation, cont),
            c.get_id(), priority, HPX_FORWARD(Ts, vs)...);
    }

    template <typename Action, typename Continuation, typename DistPolicy,
        typename... Ts>
    std::enable_if_t<traits::is_continuation<Continuation>::value &&
            traits::is_distribution_policy_v<DistPolicy>,
        bool>
    post_p(Continuation&& c, DistPolicy const& policy,
        threads::thread_priority priority, Ts&&... vs)
    {
        return policy.template apply<Action>(
            HPX_FORWARD(Continuation, c), priority, HPX_FORWARD(Ts, vs)...);
    }

    namespace detail {

        template <typename Continuation>
        struct post_dispatch<Continuation,
            std::enable_if_t<traits::is_continuation<Continuation>::value>>
        {
            template <typename Component, typename Signature, typename Derived,
                typename... Ts>
            HPX_FORCEINLINE static bool call(Continuation&& c,
                hpx::actions::basic_action<Component, Signature, Derived>,
                hpx::id_type const& id, Ts&&... ts)
            {
                return hpx::post_p<Derived>(HPX_FORWARD(Continuation, c), id,
                    actions::action_priority<Derived>(),
                    HPX_FORWARD(Ts, ts)...);
            }

            template <typename Continuation_, typename Component,
                typename Signature, typename Derived, typename Client,
                typename Stub, typename Data, typename... Ts>
            HPX_FORCEINLINE static bool call(Continuation_&& cont,
                hpx::actions::basic_action<Component, Signature, Derived>,
                components::client_base<Client, Stub, Data> const& c,
                Ts&&... ts)
            {
                // make sure the action is compatible with the component type
                typedef typename components::client_base<Client, Stub,
                    Data>::server_component_type component_type;

                static_assert(
                    traits::is_valid_action_v<Derived, component_type>,
                    "The action to invoke is not supported by the target");

                return hpx::post_p<Derived>(HPX_FORWARD(Continuation, cont),
                    c.get_id(), actions::action_priority<Derived>(),
                    HPX_FORWARD(Ts, ts)...);
            }

            template <typename Component, typename Signature, typename Derived,
                typename DistPolicy, typename... Ts>
            HPX_FORCEINLINE static std::enable_if_t<
                traits::is_distribution_policy_v<DistPolicy>, bool>
            call(Continuation&& c,
                hpx::actions::basic_action<Component, Signature, Derived>,
                DistPolicy const& policy, Ts&&... ts)
            {
                return hpx::post_p<Derived>(HPX_FORWARD(Continuation, c),
                    policy, actions::action_priority<Derived>(),
                    HPX_FORWARD(Ts, ts)...);
            }
        };
    }    // namespace detail

    template <typename Action, typename Continuation, typename... Ts>
    inline std::enable_if_t<traits::is_continuation<Continuation>::value, bool>
    post(Continuation&& c, hpx::id_type const& gid, Ts&&... vs)
    {
        return hpx::post_p<Action>(HPX_FORWARD(Continuation, c), gid,
            actions::action_priority<Action>(), HPX_FORWARD(Ts, vs)...);
    }

    template <typename Action, typename Continuation, typename Client,
        typename Stub, typename Data, typename... Ts>
    inline std::enable_if_t<traits::is_continuation<Continuation>::value, bool>
    post(Continuation&& cont,
        components::client_base<Client, Stub, Data> const& c, Ts&&... vs)
    {
        // make sure the action is compatible with the component type
        typedef typename components::client_base<Client, Stub,
            Data>::server_component_type component_type;

        static_assert(traits::is_valid_action_v<Action, component_type>,
            "The action to invoke is not supported by the target");

        return hpx::post_p<Action>(HPX_FORWARD(Continuation, cont), c.get_id(),
            actions::action_priority<Action>(), HPX_FORWARD(Ts, vs)...);
    }

    template <typename Action, typename Continuation, typename DistPolicy,
        typename... Ts>
    std::enable_if_t<traits::is_distribution_policy_v<DistPolicy> &&
            traits::is_continuation<Continuation>::value,
        bool>
    post(Continuation&& c, DistPolicy const& policy, Ts&&... vs)
    {
        return hpx::post_p<Action>(HPX_FORWARD(Continuation, c), policy,
            actions::action_priority<Action>(), HPX_FORWARD(Ts, vs)...);
    }

    ///////////////////////////////////////////////////////////////////////////
#if defined(HPX_HAVE_NETWORKING)
    namespace detail {

        template <typename Action, typename... Ts>
        inline bool post_c_p(naming::address&& addr,
            hpx::id_type const& contgid, hpx::id_type const& gid,
            threads::thread_priority priority, Ts&&... vs)
        {
            typedef
                typename hpx::traits::extract_action<Action>::local_result_type
                    local_result_type;
            typedef
                typename hpx::traits::extract_action<Action>::remote_result_type
                    remote_result_type;

            return post_r_p<Action>(HPX_MOVE(addr),
                actions::typed_continuation<local_result_type,
                    remote_result_type>(contgid),
                gid, priority, HPX_FORWARD(Ts, vs)...);
        }

        template <typename Action, typename... Ts>
        inline bool post_c(naming::address&& addr, hpx::id_type const& contgid,
            hpx::id_type const& gid, Ts&&... vs)
        {
            typedef
                typename hpx::traits::extract_action<Action>::local_result_type
                    local_result_type;
            typedef
                typename hpx::traits::extract_action<Action>::remote_result_type
                    remote_result_type;

            return post_r_p<Action>(HPX_MOVE(addr),
                actions::typed_continuation<local_result_type,
                    remote_result_type>(contgid),
                gid, actions::action_priority<Action>(),
                HPX_FORWARD(Ts, vs)...);
        }
    }    // namespace detail
#endif

    ///////////////////////////////////////////////////////////////////////////
    template <typename Action, typename... Ts>
    inline bool post_c_p(hpx::id_type const& contgid, hpx::id_type const& gid,
        threads::thread_priority priority, Ts&&... vs)
    {
        typedef typename hpx::traits::extract_action<Action>::local_result_type
            local_result_type;
        typedef typename hpx::traits::extract_action<Action>::remote_result_type
            remote_result_type;

        return hpx::post_p<Action>(
            actions::typed_continuation<local_result_type, remote_result_type>(
                contgid),
            gid, priority, HPX_FORWARD(Ts, vs)...);
    }

    template <typename Action, typename... Ts>
    inline bool post_c(
        hpx::id_type const& contgid, hpx::id_type const& gid, Ts&&... vs)
    {
        typedef typename hpx::traits::extract_action<Action>::local_result_type
            local_result_type;
        typedef typename hpx::traits::extract_action<Action>::remote_result_type
            remote_result_type;

        return hpx::post_p<Action>(
            actions::typed_continuation<local_result_type, remote_result_type>(
                contgid),
            gid, actions::action_priority<Action>(), HPX_FORWARD(Ts, vs)...);
    }

    template <typename Component, typename Signature, typename Derived,
        typename... Ts>
    inline bool post_c(
        hpx::actions::basic_action<Component, Signature, Derived> /*act*/,
        hpx::id_type const& contgid, hpx::id_type const& gid, Ts&&... vs)
    {
        typedef typename hpx::traits::extract_action<Derived>::local_result_type
            local_result_type;
        typedef
            typename hpx::traits::extract_action<Derived>::remote_result_type
                remote_result_type;

        return hpx::post_p<Derived>(
            actions::typed_continuation<local_result_type, remote_result_type>(
                contgid),
            gid, actions::action_priority<Derived>(), HPX_FORWARD(Ts, vs)...);
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename Action, typename... Ts>
    HPX_DEPRECATED_V(1, 9, "hpx::apply is deprecated, use hpx::post instead")
    inline bool apply(Ts&&... ts)
    {
        return hpx::post<Action>(HPX_FORWARD(Ts, ts)...);
    }

    template <typename... Ts>
    HPX_DEPRECATED_V(1, 9, "hpx::apply is deprecated, use hpx::post instead")
    inline bool apply(Ts&&... ts)
    {
        return hpx::post(HPX_FORWARD(Ts, ts)...);
    }

    template <typename Action, typename... Ts>
    HPX_DEPRECATED_V(
        1, 9, "hpx::apply_p is deprecated, use hpx::post_p instead")
    inline bool apply_p(Ts&&... ts)
    {
        return hpx::post_p<Action>(HPX_FORWARD(Ts, ts)...);
    }

    template <typename... Ts>
    HPX_DEPRECATED_V(
        1, 9, "hpx::apply_p is deprecated, use hpx::post_p instead")
    inline bool apply_p(Ts&&... ts)
    {
        return hpx::post_p(HPX_FORWARD(Ts, ts)...);
    }

    template <typename Action, typename... Ts>
    HPX_DEPRECATED_V(
        1, 9, "hpx::apply_c is deprecated, use hpx::post_c instead")
    inline bool apply_c(Ts&&... ts)
    {
        return hpx::post_c<Action>(HPX_FORWARD(Ts, ts)...);
    }

    template <typename... Ts>
    HPX_DEPRECATED_V(
        1, 9, "hpx::apply_c is deprecated, use hpx::post_c instead")
    inline bool apply_c(Ts&&... ts)
    {
        return hpx::post_c(HPX_FORWARD(Ts, ts)...);
    }

    template <typename Action, typename... Ts>
    HPX_DEPRECATED_V(
        1, 9, "hpx::apply_c_p is deprecated, use hpx::post_c_p instead")
    inline bool apply_c_p(Ts&&... ts)
    {
        return hpx::post_c_p<Action>(HPX_FORWARD(Ts, ts)...);
    }

    template <typename... Ts>
    HPX_DEPRECATED_V(
        1, 9, "hpx::apply_c_p is deprecated, use hpx::post_c_p instead")
    inline bool apply_c_p(Ts&&... ts)
    {
        return hpx::post_c_p(HPX_FORWARD(Ts, ts)...);
    }
}    // namespace hpx

// these files are intentionally #included last as it refers to functions
// defined here
#include <hpx/async_distributed/detail/post_implementations.hpp>
#include <hpx/async_distributed/put_parcel.hpp>
