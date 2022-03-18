//  Copyright (c) 2007-2021 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Lelbach
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/actions/apply_helper.hpp>
#include <hpx/actions_base/action_priority.hpp>
#include <hpx/actions_base/actions_base_support.hpp>
#include <hpx/actions_base/traits/extract_action.hpp>
#include <hpx/actions_base/traits/is_continuation.hpp>
#include <hpx/actions_base/traits/is_distribution_policy.hpp>
#include <hpx/actions_base/traits/is_valid_action.hpp>
#include <hpx/assert.hpp>
#include <hpx/async_distributed/applier/detail/apply_implementations_fwd.hpp>
#include <hpx/async_distributed/continuation.hpp>
#include <hpx/async_distributed/put_parcel_fwd.hpp>
#include <hpx/async_local/apply.hpp>
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

// FIXME: Error codes?

namespace hpx {
    ///////////////////////////////////////////////////////////////////////////
    // Invoked by a running HPX-thread to apply an action to any resource
    namespace applier { namespace detail {
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
        inline bool apply_r_p(naming::address&& addr, hpx::id_type const& id,
            threads::thread_priority priority, Ts&&... vs)
        {
            // If remote, create a new parcel to be sent to the destination
            // Create a new parcel with the gid, action, and arguments
            return detail::put_parcel<Action>(
                id, HPX_MOVE(addr), priority, HPX_FORWARD(Ts, vs)...);
        }

        template <typename Action, typename... Ts>
        inline bool apply_r(
            naming::address&& addr, hpx::id_type const& gid, Ts&&... vs)
        {
            return apply_r_p<Action>(HPX_MOVE(addr), gid,
                actions::action_priority<Action>(), HPX_FORWARD(Ts, vs)...);
        }
#endif

        // We know it is local and has to be directly executed.
        template <typename Action, typename... Ts>
        inline bool apply_l_p(hpx::id_type const& target,
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
            apply_helper<action_type>::call(HPX_MOVE(data), target,
                addr.address_, addr.type_, priority, HPX_FORWARD(Ts, vs)...);
            return true;    // no parcel has been sent (dest is local)
        }

        // same as above, but taking all arguments by value
        template <typename Action, typename... Ts>
        inline bool apply_l_p_val(hpx::id_type const& target,
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
            apply_helper<action_type>::call(HPX_MOVE(data), target,
                addr.address_, addr.type_, priority, HPX_MOVE(vs)...);
            return true;    // no parcel has been sent (dest is local)
        }

        template <typename Action, typename... Ts>
        inline bool apply_l(
            hpx::id_type const& target, naming::address&& addr, Ts&&... vs)
        {
            return apply_l_p<Action>(target, HPX_MOVE(addr),
                actions::action_priority<Action>(), HPX_FORWARD(Ts, vs)...);
        }
    }}    // namespace applier::detail

    ///////////////////////////////////////////////////////////////////////////
    template <typename Action, typename... Ts>
    inline bool apply_p(
        hpx::id_type const& id, threads::thread_priority priority, Ts&&... vs)
    {
        return hpx::detail::apply_impl<Action>(
            id, priority, HPX_FORWARD(Ts, vs)...);
    }

    template <typename Action, typename Client, typename Stub, typename... Ts>
    inline bool apply_p(components::client_base<Client, Stub> const& c,
        threads::thread_priority priority, Ts&&... vs)
    {
        // make sure the action is compatible with the component type
        typedef typename components::client_base<Client,
            Stub>::server_component_type component_type;

        typedef traits::is_valid_action<Action, component_type> is_valid;
        static_assert(is_valid::value,
            "The action to invoke is not supported by the target");

        return hpx::detail::apply_impl<Action>(
            c.get_id(), priority, HPX_FORWARD(Ts, vs)...);
    }

    template <typename Action, typename DistPolicy, typename... Ts>
    inline typename std::enable_if<
        traits::is_distribution_policy<DistPolicy>::value, bool>::type
    apply_p(
        DistPolicy const& policy, threads::thread_priority priority, Ts&&... vs)
    {
        return policy.template apply<Action>(priority, HPX_FORWARD(Ts, vs)...);
    }

    namespace detail {
        template <typename Action>
        struct apply_dispatch<Action,
            typename std::enable_if<traits::is_action<Action>::value>::type>
        {
            template <typename Component, typename Signature, typename Derived,
                typename... Ts>
            HPX_FORCEINLINE static bool call(
                hpx::actions::basic_action<Component, Signature, Derived>,
                hpx::id_type const& id, Ts&&... ts)
            {
                return apply_p<Derived>(id, actions::action_priority<Derived>(),
                    HPX_FORWARD(Ts, ts)...);
            }

            template <typename Component, typename Signature, typename Derived,
                typename Client, typename Stub, typename... Ts>
            HPX_FORCEINLINE static bool call(
                hpx::actions::basic_action<Component, Signature, Derived>,
                components::client_base<Client, Stub> const& c, Ts&&... ts)
            {
                // make sure the action is compatible with the component type
                typedef typename components::client_base<Client,
                    Stub>::server_component_type component_type;

                typedef traits::is_valid_action<Derived, component_type>
                    is_valid;
                static_assert(is_valid::value,
                    "The action to invoke is not supported by the target");

                return apply_p<Derived>(c.get_id(),
                    actions::action_priority<Derived>(),
                    HPX_FORWARD(Ts, ts)...);
            }

            template <typename Component, typename Signature, typename Derived,
                typename DistPolicy, typename... Ts>
            HPX_FORCEINLINE static typename std::enable_if<
                traits::is_distribution_policy<DistPolicy>::value, bool>::type
            call(hpx::actions::basic_action<Component, Signature, Derived>,
                DistPolicy const& policy, Ts&&... ts)
            {
                return apply_p<Derived>(policy,
                    actions::action_priority<Derived>(),
                    HPX_FORWARD(Ts, ts)...);
            }
        };
    }    // namespace detail

    template <typename Action, typename... Ts>
    inline bool apply(hpx::id_type const& id, Ts&&... vs)
    {
        return apply_p<Action>(
            id, actions::action_priority<Action>(), HPX_FORWARD(Ts, vs)...);
    }

    template <typename Action, typename Client, typename Stub, typename... Ts>
    inline bool apply(
        components::client_base<Client, Stub> const& c, Ts&&... vs)
    {
        // make sure the action is compatible with the component type
        typedef typename components::client_base<Client,
            Stub>::server_component_type component_type;

        typedef traits::is_valid_action<Action, component_type> is_valid;
        static_assert(is_valid::value,
            "The action to invoke is not supported by the target");

        return apply_p<Action>(c.get_id(), actions::action_priority<Action>(),
            HPX_FORWARD(Ts, vs)...);
    }

    template <typename Action, typename DistPolicy, typename... Ts>
    inline typename std::enable_if<
        traits::is_distribution_policy<DistPolicy>::value, bool>::type
    apply(DistPolicy const& policy, Ts&&... vs)
    {
        return apply_p<Action>(
            policy, actions::action_priority<Action>(), HPX_FORWARD(Ts, vs)...);
    }

    ///////////////////////////////////////////////////////////////////////////
    namespace applier { namespace detail {
#if defined(HPX_HAVE_NETWORKING)
        template <typename Action, typename Continuation, typename... Ts>
        inline bool apply_r_p(naming::address&& addr, Continuation&& c,
            hpx::id_type const& id, threads::thread_priority priority,
            Ts&&... vs)
        {
            // If remote, create a new parcel to be sent to the destination
            // Create a new parcel with the gid, action, and arguments
            return detail::put_parcel_cont<Action>(id, HPX_MOVE(addr), priority,
                HPX_FORWARD(Continuation, c), HPX_FORWARD(Ts, vs)...);
        }

        template <typename Action, typename Continuation, typename... Ts>
        inline typename std::enable_if<
            traits::is_continuation<Continuation>::value, bool>::type
        apply_r(naming::address&& addr, Continuation&& c,
            hpx::id_type const& gid, Ts&&... vs)
        {
            return apply_r_p<Action>(HPX_MOVE(addr),
                HPX_FORWARD(Continuation, c), gid,
                actions::action_priority<Action>(), HPX_FORWARD(Ts, vs)...);
        }

        template <typename Action>
        inline bool apply_r_sync_p(naming::address&& addr,
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
        inline bool apply_r_sync(
            naming::address&& addr, hpx::id_type const& gid)
        {
            return apply_r_sync_p<Action>(
                HPX_MOVE(addr), gid, actions::action_priority<Action>());
        }
#endif

        // We know it is local and has to be directly executed.
        template <typename Action, typename Continuation, typename... Ts>
        inline bool apply_l_p(Continuation&& cont, hpx::id_type const& target,
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
            apply_helper<action_type>::call(HPX_MOVE(data),
                HPX_FORWARD(Continuation, cont), target, addr.address_,
                addr.type_, priority, HPX_FORWARD(Ts, vs)...);
            return true;    // no parcel has been sent (dest is local)
        }

        template <typename Action, typename Continuation, typename... Ts>
        inline typename std::enable_if<
            traits::is_continuation<Continuation>::value, bool>::type
        apply_l(Continuation&& c, hpx::id_type const& target,
            naming::address& addr, Ts&&... vs)
        {
            return apply_l_p<Action>(HPX_FORWARD(Continuation, c), target,
                HPX_MOVE(addr), actions::action_priority<Action>(),
                HPX_FORWARD(Ts, vs)...);
        }
    }}    // namespace applier::detail

    ///////////////////////////////////////////////////////////////////////////
    template <typename Action, typename Continuation, typename... Ts>
    inline typename std::enable_if<traits::is_continuation<Continuation>::value,
        bool>::type
    apply_p(Continuation&& c, hpx::id_type const& gid,
        threads::thread_priority priority, Ts&&... vs)
    {
        return hpx::detail::apply_impl<Action>(HPX_FORWARD(Continuation, c),
            gid, priority, HPX_FORWARD(Ts, vs)...);
    }

    template <typename Action, typename Continuation, typename Client,
        typename Stub, typename... Ts>
    inline typename std::enable_if<traits::is_continuation<Continuation>::value,
        bool>::type
    apply_p(Continuation&& cont, components::client_base<Client, Stub> const& c,
        threads::thread_priority priority, Ts&&... vs)
    {
        // make sure the action is compatible with the component type
        typedef typename components::client_base<Client,
            Stub>::server_component_type component_type;

        typedef traits::is_valid_action<Action, component_type> is_valid;
        static_assert(is_valid::value,
            "The action to invoke is not supported by the target");

        return hpx::detail::apply_impl<Action>(HPX_FORWARD(Continuation, cont),
            c.get_id(), priority, HPX_FORWARD(Ts, vs)...);
    }

    template <typename Action, typename Continuation, typename DistPolicy,
        typename... Ts>
    inline
        typename std::enable_if<traits::is_continuation<Continuation>::value &&
                traits::is_distribution_policy<DistPolicy>::value,
            bool>::type
        apply_p(Continuation&& c, DistPolicy const& policy,
            threads::thread_priority priority, Ts&&... vs)
    {
        return policy.template apply<Action>(
            HPX_FORWARD(Continuation, c), priority, HPX_FORWARD(Ts, vs)...);
    }

    namespace detail {
        template <typename Continuation>
        struct apply_dispatch<Continuation,
            typename std::enable_if<
                traits::is_continuation<Continuation>::value>::type>
        {
            template <typename Component, typename Signature, typename Derived,
                typename... Ts>
            HPX_FORCEINLINE static bool call(Continuation&& c,
                hpx::actions::basic_action<Component, Signature, Derived>,
                hpx::id_type const& id, Ts&&... ts)
            {
                return apply_p<Derived>(HPX_FORWARD(Continuation, c), id,
                    actions::action_priority<Derived>(),
                    HPX_FORWARD(Ts, ts)...);
            }

            template <typename Continuation_, typename Component,
                typename Signature, typename Derived, typename Client,
                typename Stub, typename... Ts>
            HPX_FORCEINLINE static bool call(Continuation_&& cont,
                hpx::actions::basic_action<Component, Signature, Derived>,
                components::client_base<Client, Stub> const& c, Ts&&... ts)
            {
                // make sure the action is compatible with the component type
                typedef typename components::client_base<Client,
                    Stub>::server_component_type component_type;

                typedef traits::is_valid_action<Derived, component_type>
                    is_valid;
                static_assert(is_valid::value,
                    "The action to invoke is not supported by the target");

                return apply_p<Derived>(HPX_FORWARD(Continuation, cont),
                    c.get_id(), actions::action_priority<Derived>(),
                    HPX_FORWARD(Ts, ts)...);
            }

            template <typename Component, typename Signature, typename Derived,
                typename DistPolicy, typename... Ts>
            HPX_FORCEINLINE static typename std::enable_if<
                traits::is_distribution_policy<DistPolicy>::value, bool>::type
            call(Continuation&& c,
                hpx::actions::basic_action<Component, Signature, Derived>,
                DistPolicy const& policy, Ts&&... ts)
            {
                return apply_p<Derived>(HPX_FORWARD(Continuation, c), policy,
                    actions::action_priority<Derived>(),
                    HPX_FORWARD(Ts, ts)...);
            }
        };
    }    // namespace detail

    template <typename Action, typename Continuation, typename... Ts>
    inline typename std::enable_if<traits::is_continuation<Continuation>::value,
        bool>::type
    apply(Continuation&& c, hpx::id_type const& gid, Ts&&... vs)
    {
        return apply_p<Action>(HPX_FORWARD(Continuation, c), gid,
            actions::action_priority<Action>(), HPX_FORWARD(Ts, vs)...);
    }

    template <typename Action, typename Continuation, typename Client,
        typename Stub, typename... Ts>
    inline typename std::enable_if<traits::is_continuation<Continuation>::value,
        bool>::type
    apply(Continuation&& cont, components::client_base<Client, Stub> const& c,
        Ts&&... vs)
    {
        // make sure the action is compatible with the component type
        typedef typename components::client_base<Client,
            Stub>::server_component_type component_type;

        typedef traits::is_valid_action<Action, component_type> is_valid;
        static_assert(is_valid::value,
            "The action to invoke is not supported by the target");

        return apply_p<Action>(HPX_FORWARD(Continuation, cont), c.get_id(),
            actions::action_priority<Action>(), HPX_FORWARD(Ts, vs)...);
    }

    template <typename Action, typename Continuation, typename DistPolicy,
        typename... Ts>
    inline typename std::enable_if<
        traits::is_distribution_policy<DistPolicy>::value &&
            traits::is_continuation<Continuation>::value,
        bool>::type
    apply(Continuation&& c, DistPolicy const& policy, Ts&&... vs)
    {
        return apply_p<Action>(HPX_FORWARD(Continuation, c), policy,
            actions::action_priority<Action>(), HPX_FORWARD(Ts, vs)...);
    }

    ///////////////////////////////////////////////////////////////////////////
#if defined(HPX_HAVE_NETWORKING)
    namespace applier { namespace detail {
        template <typename Action, typename... Ts>
        inline bool apply_c_p(naming::address&& addr,
            hpx::id_type const& contgid, hpx::id_type const& gid,
            threads::thread_priority priority, Ts&&... vs)
        {
            typedef
                typename hpx::traits::extract_action<Action>::local_result_type
                    local_result_type;
            typedef
                typename hpx::traits::extract_action<Action>::remote_result_type
                    remote_result_type;

            return apply_r_p<Action>(HPX_MOVE(addr),
                actions::typed_continuation<local_result_type,
                    remote_result_type>(contgid),
                gid, priority, HPX_FORWARD(Ts, vs)...);
        }

        template <typename Action, typename... Ts>
        inline bool apply_c(naming::address&& addr, hpx::id_type const& contgid,
            hpx::id_type const& gid, Ts&&... vs)
        {
            typedef
                typename hpx::traits::extract_action<Action>::local_result_type
                    local_result_type;
            typedef
                typename hpx::traits::extract_action<Action>::remote_result_type
                    remote_result_type;

            return apply_r_p<Action>(HPX_MOVE(addr),
                actions::typed_continuation<local_result_type,
                    remote_result_type>(contgid),
                gid, actions::action_priority<Action>(),
                HPX_FORWARD(Ts, vs)...);
        }
    }}    // namespace applier::detail
#endif

    ///////////////////////////////////////////////////////////////////////////
    template <typename Action, typename... Ts>
    inline bool apply_c_p(hpx::id_type const& contgid, hpx::id_type const& gid,
        threads::thread_priority priority, Ts&&... vs)
    {
        typedef typename hpx::traits::extract_action<Action>::local_result_type
            local_result_type;
        typedef typename hpx::traits::extract_action<Action>::remote_result_type
            remote_result_type;

        return apply_p<Action>(
            actions::typed_continuation<local_result_type, remote_result_type>(
                contgid),
            gid, priority, HPX_FORWARD(Ts, vs)...);
    }

    template <typename Action, typename... Ts>
    inline bool apply_c(
        hpx::id_type const& contgid, hpx::id_type const& gid, Ts&&... vs)
    {
        typedef typename hpx::traits::extract_action<Action>::local_result_type
            local_result_type;
        typedef typename hpx::traits::extract_action<Action>::remote_result_type
            remote_result_type;

        return apply_p<Action>(
            actions::typed_continuation<local_result_type, remote_result_type>(
                contgid),
            gid, actions::action_priority<Action>(), HPX_FORWARD(Ts, vs)...);
    }

    template <typename Component, typename Signature, typename Derived,
        typename... Ts>
    inline bool apply_c(
        hpx::actions::basic_action<Component, Signature, Derived> /*act*/,
        hpx::id_type const& contgid, hpx::id_type const& gid, Ts&&... vs)
    {
        typedef typename hpx::traits::extract_action<Derived>::local_result_type
            local_result_type;
        typedef
            typename hpx::traits::extract_action<Derived>::remote_result_type
                remote_result_type;

        return apply_p<Derived>(
            actions::typed_continuation<local_result_type, remote_result_type>(
                contgid),
            gid, actions::action_priority<Derived>(), HPX_FORWARD(Ts, vs)...);
    }
}    // namespace hpx

// these files are intentionally #included last as it refers to functions
// defined here
#include <hpx/async_distributed/applier/detail/apply_implementations.hpp>
#include <hpx/async_distributed/put_parcel.hpp>
