//  Copyright (c) 2007-2016 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_APPLIER_APPLY_NOV_27_2008_0957AM)
#define HPX_APPLIER_APPLY_NOV_27_2008_0957AM

#include <hpx/config.hpp>
#include <hpx/runtime.hpp>
#include <hpx/runtime/actions/action_priority.hpp>
#include <hpx/runtime/agas/interface.hpp>
#include <hpx/runtime/applier/apply_helper.hpp>
#include <hpx/runtime/applier/detail/apply_implementations.hpp>
#include <hpx/runtime/components/client_base.hpp>
#include <hpx/runtime/naming/address.hpp>
#include <hpx/runtime/naming/id_type.hpp>
#include <hpx/runtime/naming/name.hpp>
#include <hpx/runtime/parcelset/put_parcel.hpp>
#include <hpx/runtime/parcelset/detail/parcel_await.hpp>
#include <hpx/traits/component_type_is_compatible.hpp>
#include <hpx/traits/extract_action.hpp>
#include <hpx/traits/is_action.hpp>
#include <hpx/traits/is_continuation.hpp>
#include <hpx/traits/is_distribution_policy.hpp>
#include <hpx/traits/is_valid_action.hpp>

#include <type_traits>
#include <utility>

// FIXME: Error codes?

namespace hpx
{
    ///////////////////////////////////////////////////////////////////////////
    // Invoked by a running HPX-thread to apply an action to any resource
    namespace applier { namespace detail
    {
        ///////////////////////////////////////////////////////////////////////
        template <typename Action>
        inline naming::address&& complement_addr(naming::address& addr)
        {
            if (components::component_invalid == addr.type_)
            {
                addr.type_ = components::get_component_type<
                    typename Action::component_type>();
            }
            return std::move(addr);
        }

        template <typename Action, typename ...Ts>
        inline bool
        put_parcel(naming::id_type const& id, naming::address&& addr,
            threads::thread_priority priority, Ts&&... vs)
        {
            typedef
                typename hpx::traits::extract_action<Action>::type
                action_type;
            action_type act;

            parcelset::put_parcel(id, complement_addr<action_type>(addr),
                act, priority, std::forward<Ts>(vs)...);

            return false;     // destinations are remote
        }

        template <typename Action, typename Continuation, typename ...Ts>
        inline bool
        put_parcel_cont(naming::id_type const& id, naming::address&& addr,
            threads::thread_priority priority,
            Continuation && cont, Ts&&... vs)
        {
            typedef
                typename hpx::traits::extract_action<Action>::type
                action_type;
            action_type act;

            parcelset::put_parcel(id, complement_addr<action_type>(addr),
                std::forward<Continuation>(cont),
                act, priority, std::forward<Ts>(vs)...);

            return false;     // destinations are remote
        }

        template <typename Action, typename ...Ts>
        inline bool
        put_parcel_cb(naming::id_type const& id, naming::address&& addr,
            threads::thread_priority priority,
            parcelset::parcelhandler::write_handler_type const& cb, Ts&&... vs)
        {
            typedef
                typename hpx::traits::extract_action<Action>::type
                action_type;
            action_type act;

            parcelset::put_parcel_cb(cb, id,
                complement_addr<action_type>(addr),
                act, priority, std::forward<Ts>(vs)...);

            return false;     // destinations are remote
        }

        template <typename Action, typename ...Ts>
        inline bool
        put_parcel_cb(naming::id_type const& id, naming::address&& addr,
            threads::thread_priority priority,
            parcelset::parcelhandler::write_handler_type && cb, Ts&&... vs)
        {
            typedef
                typename hpx::traits::extract_action<Action>::type
                action_type;
            action_type act;

            parcelset::put_parcel_cb(std::move(cb), id,
                complement_addr<action_type>(addr),
                act, priority, std::forward<Ts>(vs)...);

            return false;     // destinations are remote
        }

        template <typename Action, typename Continuation, typename ...Ts>
        inline bool
        put_parcel_cont_cb(naming::id_type const& id,
            naming::address&& addr, threads::thread_priority priority,
            Continuation && cont,
            parcelset::parcelhandler::write_handler_type const& cb, Ts&&... vs)
        {
            typedef
                typename hpx::traits::extract_action<Action>::type
                action_type;
            action_type act;

            parcelset::put_parcel_cb(cb, id,
                complement_addr<action_type>(addr),
                std::forward<Continuation>(cont),
                act, priority, std::forward<Ts>(vs)...);

            return false;     // destinations are remote
        }

        template <typename Action, typename Continuation, typename ...Ts>
        inline bool
        put_parcel_cont_cb(naming::id_type const& id,
            naming::address&& addr, threads::thread_priority priority,
            Continuation && cont,
            parcelset::parcelhandler::write_handler_type && cb, Ts&&... vs)
        {
            typedef
                typename hpx::traits::extract_action<Action>::type
                action_type;
            action_type act;

            parcelset::put_parcel_cb(std::move(cb), id,
                complement_addr<action_type>(addr),
                std::forward<Continuation>(cont),
                act, priority, std::forward<Ts>(vs)...);

            return false;     // destinations are remote
        }

        // We know it is remote.
        template <typename Action, typename ...Ts>
        inline bool
        apply_r_p(naming::address&& addr, naming::id_type const& id,
            threads::thread_priority priority, Ts&&... vs)
        {
            // If remote, create a new parcel to be sent to the destination
            // Create a new parcel with the gid, action, and arguments
            return detail::put_parcel<Action>(id, std::move(addr), priority,
                std::forward<Ts>(vs)...);
        }

        template <typename Action, typename ...Ts>
        inline bool
        apply_r (naming::address&& addr, naming::id_type const& gid,
            Ts&&... vs)
        {
            return apply_r_p<Action>(std::move(addr), gid,
                actions::action_priority<Action>(), std::forward<Ts>(vs)...);
        }

        // We know it is local and has to be directly executed.
        template <typename Action, typename ...Ts>
        inline bool
        apply_l_p(naming::id_type const& target, naming::address&& addr,
            threads::thread_priority priority, Ts&&... vs)
        {
            typedef typename hpx::traits::extract_action<Action>::type action_type;

            HPX_ASSERT(traits::component_type_is_compatible<
                typename action_type::component_type>::call(addr));

            threads::thread_init_data data;
            apply_helper<action_type>::call(std::move(data), target,
                addr.address_, addr.type_, priority, std::forward<Ts>(vs)...);
            return true;     // no parcel has been sent (dest is local)
        }

        // same as above, but taking all arguments by value
        template <typename Action, typename ...Ts>
        inline bool
        apply_l_p_val(naming::id_type const& target, naming::address&& addr,
            threads::thread_priority priority, Ts... vs)
        {
            typedef typename hpx::traits::extract_action<Action>::type action_type;

            HPX_ASSERT(traits::component_type_is_compatible<
                typename action_type::component_type>::call(addr));

            threads::thread_init_data data;
            apply_helper<action_type>::call(std::move(data), target,
                addr.address_, addr.type_, priority, std::move(vs)...);
            return true;     // no parcel has been sent (dest is local)
        }

        template <typename Action, typename ...Ts>
        inline bool
        apply_l (naming::id_type const& target, naming::address && addr,
            Ts&&... vs)
        {
            return apply_l_p<Action>(target, std::move(addr),
                actions::action_priority<Action>(),
                std::forward<Ts>(vs)...);
        }
    }}

    ///////////////////////////////////////////////////////////////////////////
    template <typename Action, typename ...Ts>
    inline bool
    apply_p(naming::id_type const& id, threads::thread_priority priority,
        Ts&&... vs)
    {
        return hpx::detail::apply_impl<Action>(id, priority,
            std::forward<Ts>(vs)...);
    }

    template <typename Action, typename Client, typename Stub, typename ...Ts>
    inline bool
    apply_p(components::client_base<Client, Stub> const& c,
        threads::thread_priority priority, Ts&&... vs)
    {
        // make sure the action is compatible with the component type
        typedef typename components::client_base<
                Client, Stub
            >::server_component_type component_type;

        typedef traits::is_valid_action<Action, component_type> is_valid;
        static_assert(is_valid::value,
            "The action to invoke is not supported by the target");

        return hpx::detail::apply_impl<Action>(
            c.get_id(), priority, std::forward<Ts>(vs)...);
    }

    template <typename Action, typename DistPolicy, typename ...Ts>
    inline typename std::enable_if<
        traits::is_distribution_policy<DistPolicy>::value, bool
    >::type
    apply_p(DistPolicy const& policy, threads::thread_priority priority,
        Ts&&... vs)
    {
        return policy.template apply<Action>(priority, std::forward<Ts>(vs)...);
    }

    namespace detail
    {
        // dispatch point used for apply implementations
        template <typename Func, typename Enable = void>
        struct apply_dispatch;

        template <typename Action>
        struct apply_dispatch<Action,
            typename std::enable_if<
                traits::is_action<Action>::value
            >::type>
        {
            template <typename Component, typename Signature, typename Derived,
                typename ...Ts>
            HPX_FORCEINLINE static bool
            call(hpx::actions::basic_action<Component, Signature, Derived>,
                naming::id_type const& id, Ts&&... ts)
            {
                return apply_p<Derived>(id,
                    actions::action_priority<Derived>(),
                    std::forward<Ts>(ts)...);
            }

            template <typename Component, typename Signature, typename Derived,
                typename Client, typename Stub, typename ...Ts>
            HPX_FORCEINLINE static bool
            call(hpx::actions::basic_action<Component, Signature, Derived>,
                components::client_base<Client, Stub> const& c, Ts&&... ts)
            {
                // make sure the action is compatible with the component type
                typedef typename components::client_base<
                        Client, Stub
                    >::server_component_type component_type;

                typedef traits::is_valid_action<Derived, component_type> is_valid;
                static_assert(is_valid::value,
                    "The action to invoke is not supported by the target");

                return apply_p<Derived>(c.get_id(),
                    actions::action_priority<Derived>(),
                    std::forward<Ts>(ts)...);
            }

            template <typename Component, typename Signature, typename Derived,
                typename DistPolicy, typename ...Ts>
            HPX_FORCEINLINE static typename std::enable_if<
                traits::is_distribution_policy<DistPolicy>::value, bool
            >::type
            call(hpx::actions::basic_action<Component, Signature, Derived>,
                DistPolicy const& policy, Ts&&... ts)
            {
                return apply_p<Derived>(policy,
                    actions::action_priority<Derived>(),
                    std::forward<Ts>(ts)...);
            }
        };
    }

    template <typename Action, typename ...Ts>
    inline bool
    apply(naming::id_type const& id, Ts&&... vs)
    {
        return apply_p<Action>(id, actions::action_priority<Action>(),
            std::forward<Ts>(vs)...);
    }

    template <typename Action, typename Client, typename Stub,
        typename ...Ts>
    inline bool
    apply(components::client_base<Client, Stub> const& c, Ts&&... vs)
    {
        // make sure the action is compatible with the component type
        typedef typename components::client_base<
                Client, Stub
            >::server_component_type component_type;

        typedef traits::is_valid_action<Action, component_type> is_valid;
        static_assert(is_valid::value,
            "The action to invoke is not supported by the target");

        return apply_p<Action>(c.get_id(),
            actions::action_priority<Action>(), std::forward<Ts>(vs)...);
    }

    template <typename Action, typename DistPolicy, typename ...Ts>
    inline typename std::enable_if<
        traits::is_distribution_policy<DistPolicy>::value, bool
    >::type
    apply(DistPolicy const& policy, Ts&&... vs)
    {
        return apply_p<Action>(policy, actions::action_priority<Action>(),
            std::forward<Ts>(vs)...);
    }

    ///////////////////////////////////////////////////////////////////////////
    namespace applier { namespace detail
    {
        template <typename Action, typename Continuation, typename ...Ts>
        inline bool
        apply_r_p(naming::address&& addr, Continuation && c,
            naming::id_type const& id, threads::thread_priority priority,
            Ts&&... vs)
        {
            // If remote, create a new parcel to be sent to the destination
            // Create a new parcel with the gid, action, and arguments
            return detail::put_parcel_cont<Action>(id, std::move(addr),
                priority, std::forward<Continuation>(c), std::forward<Ts>(vs)...);
        }

        template <typename Action, typename Continuation, typename ...Ts>
        inline typename std::enable_if<
            traits::is_continuation<Continuation>::value, bool
        >::type
        apply_r (naming::address&& addr, Continuation && c,
            naming::id_type const& gid, Ts&&... vs)
        {
            return apply_r_p<Action>(std::move(addr), std::forward<Continuation>(c), gid,
                actions::action_priority<Action>(),
                std::forward<Ts>(vs)...);
        }

        template <typename Action>
        inline bool
        apply_r_sync_p(naming::address&& addr, naming::id_type const& id,
            threads::thread_priority priority)
        {
            typedef typename hpx::traits::extract_action<Action>::type action_type_;

            // If remote, create a new parcel to be sent to the destination
            // Create a new parcel with the gid, action, and arguments
            // Send the parcel through the parcel handler
            HPX_ASSERT(id.get_management_type() == naming::id_type::unmanaged);
            naming::gid_type gid = id.get_gid();
            parcelset::parcel p =
                parcelset::detail::create_parcel::call(std::false_type(),
                    std::move(gid), complement_addr<action_type_>(addr),
                    action_type_(), priority
                );

            parcelset::detail::parcel_await(std::move(p),
                parcelset::write_handler_type(), 0,
                [](parcelset::parcel&& p, parcelset::write_handler_type&&)
                {
                    hpx::get_runtime().get_parcel_handler()
                        .sync_put_parcel(std::move(p));
                }).apply();
            return false;     // destination is remote
        }

        template <typename Action>
        inline bool
        apply_r_sync(naming::address&& addr, naming::id_type const& gid)
        {
            return apply_r_sync_p<Action>(std::move(addr), gid,
                actions::action_priority<Action>());
        }

        // We know it is local and has to be directly executed.
        template <typename Action, typename Continuation, typename ...Ts>
        inline bool
        apply_l_p(Continuation && cont,
            naming::id_type const& target, naming::address&& addr,
            threads::thread_priority priority, Ts&&... vs)
        {
            typedef typename hpx::traits::extract_action<
                    Action
                >::type action_type;

            HPX_ASSERT(traits::component_type_is_compatible<
                typename action_type::component_type>::call(addr));

            threads::thread_init_data data;
            apply_helper<action_type>::call(std::move(data),
                std::forward<Continuation>(cont), target,
                addr.address_, addr.type_, priority, std::forward<Ts>(vs)...);
            return true;     // no parcel has been sent (dest is local)
        }

        template <typename Action, typename Continuation, typename ...Ts>
        inline typename std::enable_if<
            traits::is_continuation<Continuation>::value, bool
        >::type
        apply_l (Continuation && c, naming::id_type const& target,
            naming::address& addr, Ts&&... vs)
        {
            return apply_l_p<Action>(std::forward<Continuation>(c),
                target, std::move(addr),
                actions::action_priority<Action>(), std::forward<Ts>(vs)...);
        }
    }}

    ///////////////////////////////////////////////////////////////////////////
    template <typename Action, typename Continuation, typename ...Ts>
    inline typename std::enable_if<
        traits::is_continuation<Continuation>::value, bool
    >::type
    apply_p(Continuation && c, naming::id_type const& gid,
        threads::thread_priority priority, Ts&&... vs)
    {
        return hpx::detail::apply_impl<Action>(
            std::forward<Continuation>(c), gid, priority,
            std::forward<Ts>(vs)...);
    }

    template <typename Action, typename Continuation, typename Client,
        typename Stub, typename ...Ts>
    inline typename std::enable_if<
        traits::is_continuation<Continuation>::value, bool
    >::type
    apply_p(Continuation && cont,
        components::client_base<Client, Stub> const& c,
        threads::thread_priority priority, Ts&&... vs)
    {
        // make sure the action is compatible with the component type
        typedef typename components::client_base<
                Client, Stub
            >::server_component_type component_type;

        typedef traits::is_valid_action<Action, component_type> is_valid;
        static_assert(is_valid::value,
            "The action to invoke is not supported by the target");

        return hpx::detail::apply_impl<Action>(
            std::forward<Continuation>(cont), c.get_id(), priority,
            std::forward<Ts>(vs)...);
    }

    template <typename Action, typename Continuation, typename DistPolicy,
        typename ...Ts>
    inline typename std::enable_if<
        traits::is_continuation<Continuation>::value &&
            traits::is_distribution_policy<DistPolicy>::value,
        bool
    >::type
    apply_p(Continuation && c, DistPolicy const& policy,
        threads::thread_priority priority, Ts&&... vs)
    {
        return policy.template apply<Action>(
            std::forward<Continuation>(c), priority, std::forward<Ts>(vs)...);
    }

    namespace detail
    {
        template <typename Continuation>
        struct apply_dispatch<Continuation,
            typename std::enable_if<
                traits::is_continuation<Continuation>::value
            >::type>
        {
            template <typename Component, typename Signature, typename Derived,
                typename ...Ts>
            HPX_FORCEINLINE static bool
            call(Continuation && c,
                hpx::actions::basic_action<Component, Signature, Derived>,
                naming::id_type const& id, Ts&&... ts)
            {
                return apply_p<Derived>(std::forward<Continuation>(c), id,
                    actions::action_priority<Derived>(),
                    std::forward<Ts>(ts)...);
            }

            template <typename Continuation_, typename Component,
                typename Signature, typename Derived, typename Client,
                typename Stub, typename ...Ts>
            HPX_FORCEINLINE static bool
            call(Continuation_ && cont,
                hpx::actions::basic_action<Component, Signature, Derived>,
                components::client_base<Client, Stub> const& c, Ts&&... ts)
            {
                // make sure the action is compatible with the component type
                typedef typename components::client_base<
                        Client, Stub
                    >::server_component_type component_type;

                typedef traits::is_valid_action<Derived, component_type> is_valid;
                static_assert(is_valid::value,
                    "The action to invoke is not supported by the target");

                return apply_p<Derived>(std::forward<Continuation>(cont),
                    c.get_id(), actions::action_priority<Derived>(),
                    std::forward<Ts>(ts)...);
            }

            template <typename Component, typename Signature, typename Derived,
                typename DistPolicy, typename ...Ts>
            HPX_FORCEINLINE static typename std::enable_if<
                traits::is_distribution_policy<DistPolicy>::value, bool
            >::type
            call(Continuation && c,
                hpx::actions::basic_action<Component, Signature, Derived>,
                DistPolicy const& policy, Ts&&... ts)
            {
                return apply_p<Derived>(std::forward<Continuation>(c), policy,
                    actions::action_priority<Derived>(),
                    std::forward<Ts>(ts)...);
            }
        };
    }

    template <typename Action, typename Continuation, typename ...Ts>
    inline typename std::enable_if<
        traits::is_continuation<Continuation>::value, bool
    >::type
    apply(Continuation && c, naming::id_type const& gid, Ts&&... vs)
    {
        return apply_p<Action>(std::forward<Continuation>(c), gid,
            actions::action_priority<Action>(),
            std::forward<Ts>(vs)...);
    }

    template <typename Action, typename Continuation, typename Client,
        typename Stub, typename ...Ts>
    inline typename std::enable_if<
        traits::is_continuation<Continuation>::value, bool
    >::type
    apply(Continuation && cont, components::client_base<Client, Stub> const& c,
        Ts&&... vs)
    {
        // make sure the action is compatible with the component type
        typedef typename components::client_base<
                Client, Stub
            >::server_component_type component_type;

        typedef traits::is_valid_action<Action, component_type> is_valid;
        static_assert(is_valid::value,
            "The action to invoke is not supported by the target");

        return apply_p<Action>(std::forward<Continuation>(cont), c.get_id(),
            actions::action_priority<Action>(), std::forward<Ts>(vs)...);
    }

    template <typename Action, typename Continuation, typename DistPolicy,
        typename ...Ts>
    inline typename std::enable_if<
        traits::is_distribution_policy<DistPolicy>::value &&
            traits::is_continuation<Continuation>::value,
        bool
    >::type
    apply(Continuation && c, DistPolicy const& policy, Ts&&... vs)
    {
        return apply_p<Action>(std::forward<Continuation>(c), policy,
            actions::action_priority<Action>(), std::forward<Ts>(vs)...);
    }

    ///////////////////////////////////////////////////////////////////////////
    namespace applier { namespace detail
    {
        template <typename Action, typename ...Ts>
        inline bool
        apply_c_p(naming::address&& addr, naming::id_type const& contgid,
            naming::id_type const& gid, threads::thread_priority priority,
            Ts&&... vs)
        {
            typedef
                typename hpx::traits::extract_action<Action>::local_result_type
                local_result_type;
            typedef
                typename hpx::traits::extract_action<Action>::remote_result_type
                remote_result_type;

            return apply_r_p<Action>(std::move(addr),
                actions::typed_continuation<
                    local_result_type, remote_result_type>(contgid),
                gid, priority, std::forward<Ts>(vs)...);
        }

        template <typename Action, typename ...Ts>
        inline bool
        apply_c(naming::address&& addr, naming::id_type const& contgid,
            naming::id_type const& gid, Ts&&... vs)
        {
            typedef
                typename hpx::traits::extract_action<Action>::local_result_type
                local_result_type;
            typedef
                typename hpx::traits::extract_action<Action>::remote_result_type
                remote_result_type;

            return apply_r_p<Action>(std::move(addr),
                actions::typed_continuation<
                    local_result_type, remote_result_type>(contgid),
                gid, actions::action_priority<Action>(),
                std::forward<Ts>(vs)...);
        }
    }}

    ///////////////////////////////////////////////////////////////////////////
    template <typename Action, typename ...Ts>
    inline bool
    apply_c_p(naming::id_type const& contgid, naming::id_type const& gid,
        threads::thread_priority priority, Ts&&... vs)
    {
        typedef
            typename hpx::traits::extract_action<Action>::local_result_type
            local_result_type;
        typedef
            typename hpx::traits::extract_action<Action>::remote_result_type
            remote_result_type;

        return apply_p<Action>(
            actions::typed_continuation<
                local_result_type, remote_result_type
            >(contgid), gid, priority, std::forward<Ts>(vs)...);
    }

    template <typename Action, typename ...Ts>
    inline bool
    apply_c(naming::id_type const& contgid, naming::id_type const& gid,
        Ts&&... vs)
    {
        typedef
            typename hpx::traits::extract_action<Action>::local_result_type
            local_result_type;
        typedef
            typename hpx::traits::extract_action<Action>::remote_result_type
            remote_result_type;

        return apply_p<Action>(
            actions::typed_continuation<
                local_result_type, remote_result_type
            >(contgid), gid, actions::action_priority<Action>(),
            std::forward<Ts>(vs)...);
    }

    template <typename Component, typename Signature, typename Derived,
        typename ...Ts>
    inline bool
    apply_c(
        hpx::actions::basic_action<Component, Signature, Derived> /*act*/,
        naming::id_type const& contgid, naming::id_type const& gid,
        Ts&&... vs)
    {
        typedef
            typename hpx::traits::extract_action<Derived>::local_result_type
            local_result_type;
        typedef
            typename hpx::traits::extract_action<Derived>::remote_result_type
            remote_result_type;

        return apply_p<Derived>(
            actions::typed_continuation<
                local_result_type, remote_result_type
            >(contgid), gid, actions::action_priority<Derived>(),
            std::forward<Ts>(vs)...);
    }
}

#endif
