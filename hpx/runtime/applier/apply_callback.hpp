//  Copyright (c) 2007-2015 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_APPLIER_APPLY_CALLBACK_DEC_16_2012_1228PM)
#define HPX_APPLIER_APPLY_CALLBACK_DEC_16_2012_1228PM

#include <hpx/hpx_fwd.hpp>
#include <hpx/exception.hpp>

#include <hpx/runtime/applier/apply.hpp>

#include <boost/make_shared.hpp>

namespace hpx
{
    ///////////////////////////////////////////////////////////////////////////
    namespace applier { namespace detail
    {
        template <typename Action, typename Callback, typename ...Ts>
        inline bool
        apply_r_p_cb(naming::address&& addr, naming::id_type const& id,
            threads::thread_priority priority, Callback && cb, Ts&&... vs)
        {
            // If remote, create a new parcel to be sent to the destination
            // Create a new parcel with the gid, action, and arguments
            lcos::local::detail::invoke_when_ready(
                detail::put_parcel<Action>(id, std::move(addr), priority,
                    actions::continuation_type(), std::forward<Callback>(cb)),
                std::forward<Ts>(vs)...);
            return false;     // destinations are remote
        }

        template <typename Action, typename Callback, typename ...Ts>
        inline bool
        apply_r_cb(naming::address&& addr, naming::id_type const& gid,
            Callback && cb, Ts&&... vs)
        {
            return apply_r_p_cb<Action>(std::move(addr), gid,
                actions::action_priority<Action>(), std::forward<Callback>(cb),
                std::forward<Ts>(vs)...);
        }
    }}

    ///////////////////////////////////////////////////////////////////////////
    template <typename Action, typename Callback, typename ...Ts>
    inline bool
    apply_p_cb(naming::id_type const& gid, threads::thread_priority priority,
        Callback && cb, Ts&&... vs)
    {
        return hpx::detail::apply_cb_impl<Action>(
            actions::continuation_type(), gid, priority,
            std::forward<Callback>(cb), std::forward<Ts>(vs)...);
    }

    template <typename Action, typename Callback, typename ...Ts>
    inline bool
    apply_cb(naming::id_type const& gid, Callback && cb, Ts&&... vs)
    {
        return apply_p_cb<Action>(gid, actions::action_priority<Action>(),
            std::forward<Callback>(cb), std::forward<Ts>(vs)...);
    }

    template <typename Component, typename Signature, typename Derived,
        typename Callback, typename ...Ts>
    inline bool
    apply_cb(
        hpx::actions::basic_action<Component, Signature, Derived> /*act*/,
        naming::id_type const& gid, Callback && cb, Ts&&... vs)
    {
        return apply_p_cb<Derived>(gid, actions::action_priority<Derived>(),
            std::forward<Callback>(cb), std::forward<Ts>(vs)...);
    }

    template <typename Action, typename DistPolicy, typename Callback,
        typename ...Ts>
    inline typename boost::enable_if_c<
        traits::is_distribution_policy<DistPolicy>::value, bool
    >::type
    apply_p_cb(DistPolicy const& policy, threads::thread_priority priority,
        Callback && cb, Ts&&... vs)
    {
        return policy.template apply_cb<Action>(
            actions::continuation_type(), priority,
            std::forward<Callback>(cb), std::forward<Ts>(vs)...);
    }

    template <typename Action, typename DistPolicy, typename Callback,
        typename ...Ts>
    inline typename boost::enable_if_c<
        traits::is_distribution_policy<DistPolicy>::value, bool
    >::type
    apply_cb(DistPolicy const& policy, Callback && cb, Ts&&... vs)
    {
        return apply_p_cb<Action>(policy, actions::action_priority<Action>(),
            std::forward<Callback>(cb), std::forward<Ts>(vs)...);
    }

    template <typename Component, typename Signature, typename Derived,
        typename DistPolicy, typename Callback, typename ...Ts>
    inline typename boost::enable_if_c<
        traits::is_distribution_policy<DistPolicy>::value, bool
    >::type
    apply_cb(
        hpx::actions::basic_action<Component, Signature, Derived> /*act*/,
        DistPolicy const& policy, Callback && cb, Ts&&... vs)
    {
        return apply_p_cb<Derived>(policy, actions::action_priority<Derived>(),
            std::forward<Callback>(cb), std::forward<Ts>(vs)...);
    }

    ///////////////////////////////////////////////////////////////////////////
    namespace applier { namespace detail
    {
        template <typename Action, typename Callback, typename ...Ts>
        inline bool
        apply_r_p_cb(naming::address&& addr,
            actions::continuation_type const& c, naming::id_type const& id,
            threads::thread_priority priority, Callback && cb, Ts&&... vs)
        {
            // If remote, create a new parcel to be sent to the destination
            // Create a new parcel with the gid, action, and arguments
            lcos::local::detail::invoke_when_ready(
                detail::put_parcel<Action>(id, std::move(addr), priority, c,
                    std::forward<Callback>(cb)),
                std::forward<Ts>(vs)...);
            return false;     // destination is remote
        }

        template <typename Action, typename Callback, typename ...Ts>
        inline bool
        apply_r_cb(naming::address&& addr, actions::continuation_type const& c,
            naming::id_type const& gid, Callback && cb, Ts&&... vs)
        {
            return apply_r_p_cb<Action>(std::move(addr), c, gid,
                actions::action_priority<Action>(), std::forward<Callback>(cb),
                std::forward<Ts>(vs)...);
        }
    }}

    ///////////////////////////////////////////////////////////////////////////
    template <typename Action, typename Callback, typename ...Ts>
    inline bool
    apply_p_cb(actions::continuation_type const& c, naming::address&& addr,
        naming::id_type const& gid, threads::thread_priority priority,
        Callback && cb, Ts&&... vs)
    {
        if (!traits::action_is_target_valid<Action>::call(gid)) {
            HPX_THROW_EXCEPTION(bad_parameter, "apply_p_cb",
                boost::str(boost::format(
                    "the target (destination) does not match the action type (%s)"
                ) % hpx::actions::detail::get_action_name<Action>()));
            return false;
        }

        // Determine whether the gid is local or remote
        if (addr.locality_ == hpx::get_locality()) {
            // apply locally
            bool result = applier::detail::apply_l_p<Action>(c, gid,
                std::move(addr), priority, std::forward<Ts>(vs)...);

            // invoke callback
            cb(boost::system::error_code(), parcelset::parcel());
            return result;
        }

        // apply remotely
        return applier::detail::apply_r_p_cb<Action>(std::move(addr), c, gid,
            priority, std::forward<Callback>(cb), std::forward<Ts>(vs)...);
    }

    template <typename Action, typename Callback, typename ...Ts>
    inline bool
    apply_p_cb(actions::continuation_type const& c, naming::id_type const& gid,
        threads::thread_priority priority, Callback && cb, Ts&&... vs)
    {
        return hpx::detail::apply_cb_impl<Action>(c, gid, priority,
            std::forward<Callback>(cb), std::forward<Ts>(vs)...);
    }

    template <typename Action, typename Callback, typename ...Ts>
    inline bool
    apply_cb(actions::continuation_type const& c, naming::id_type const& gid,
        Callback && cb, Ts&&... vs)
    {
        return apply_p_cb<Action>(c, gid, actions::action_priority<Action>(),
            std::forward<Callback>(cb), std::forward<Ts>(vs)...);
    }

    template <typename Component, typename Signature, typename Derived,
        typename Callback, typename ...Ts>
    inline bool
    apply_cb(actions::continuation_type const& c,
        hpx::actions::basic_action<Component, Signature, Derived> /*act*/,
        naming::id_type const& gid, Callback && cb, Ts&&... vs)
    {
        return apply_p<Derived>(c, gid, actions::action_priority<Derived>(),
            std::forward<Callback>(cb), std::forward<Ts>(vs)...);
    }

    template <typename Action, typename DistPolicy, typename Callback,
        typename ...Ts>
    inline typename boost::enable_if_c<
        traits::is_distribution_policy<DistPolicy>::value, bool
    >::type
    apply_p_cb(actions::continuation_type const& c, DistPolicy const& policy,
        threads::thread_priority priority, Callback && cb, Ts&&... vs)
    {
        return policy.template apply_cb<Action>(
            c, priority, std::forward<Callback>(cb), std::forward<Ts>(vs)...);
    }

    template <typename Action, typename DistPolicy, typename Callback,
        typename ...Ts>
    inline typename boost::enable_if_c<
        traits::is_distribution_policy<DistPolicy>::value, bool
    >::type
    apply_cb(actions::continuation_type const& c, DistPolicy const& policy,
        Callback && cb, Ts&&... vs)
    {
        return apply_p_cb<Action>(c, policy, actions::action_priority<Action>(),
            std::forward<Callback>(cb), std::forward<Ts>(vs)...);
    }

    template <typename Component, typename Signature, typename Derived,
        typename DistPolicy, typename Callback, typename ...Ts>
    inline typename boost::enable_if_c<
        traits::is_distribution_policy<DistPolicy>::value, bool
    >::type
    apply_cb(actions::continuation_type const& c,
        hpx::actions::basic_action<Component, Signature, Derived> /*act*/,
        DistPolicy const& policy, Callback && cb, Ts&&... vs)
    {
        return apply_p<Derived>(c, policy, actions::action_priority<Derived>(),
            std::forward<Callback>(cb), std::forward<Ts>(vs)...);
    }

    ///////////////////////////////////////////////////////////////////////////
    namespace applier { namespace detail
    {
        template <typename Action, typename Callback, typename ...Ts>
        inline bool
        apply_c_p_cb(naming::address&& addr, naming::id_type const& contgid,
            naming::id_type const& gid, threads::thread_priority priority,
            Callback && cb, Ts&&... vs)
        {
            typedef
                typename hpx::actions::extract_action<Action>::result_type
                result_type;

            return apply_r_p_cb<Action>(std::move(addr),
                boost::make_shared<
                    actions::typed_continuation<result_type>
                >(contgid),
                gid, priority, std::forward<Callback>(cb),
                std::forward<Ts>(vs)...);
        }

        template <typename Action, typename Callback, typename ...Ts>
        inline bool
        apply_c_cb(naming::address&& addr, naming::id_type const& contgid,
            naming::id_type const& gid, Callback && cb, Ts&&... vs)
        {
            typedef
                typename hpx::actions::extract_action<Action>::result_type
                result_type;

            return apply_r_p_cb<Action>(std::move(addr),
                boost::make_shared<
                    actions::typed_continuation<result_type>
                >(contgid),
                gid, actions::action_priority<Action>(),
                std::forward<Callback>(cb),
                std::forward<Ts>(vs)...);
        }
    }}

    ///////////////////////////////////////////////////////////////////////////
    template <typename Action, typename Callback, typename ...Ts>
    inline bool
    apply_c_p_cb(naming::id_type const& contgid, naming::id_type const& gid,
        threads::thread_priority priority, Callback && cb, Ts&&... vs)
    {
        typedef
            typename hpx::actions::extract_action<Action>::result_type
            result_type;

        return apply_p_cb<Action>(
            boost::make_shared<
                actions::typed_continuation<result_type>
            >(contgid),
            gid, priority, std::forward<Callback>(cb),
            std::forward<Ts>(vs)...);
    }

    template <typename Action, typename Callback, typename ...Ts>
    inline bool
    apply_c_cb(naming::id_type const& contgid, naming::id_type const& gid,
        Callback && cb, Ts&&... vs)
    {
        typedef
            typename hpx::actions::extract_action<Action>::result_type
            result_type;

        return apply_p_cb<Action>(
            boost::make_shared<
                actions::typed_continuation<result_type>
            >(contgid),
            gid, actions::action_priority<Action>(),
            std::forward<Callback>(cb), std::forward<Ts>(vs)...);
    }

    template <typename Action, typename Callback, typename ...Ts>
    inline bool
    apply_c_p_cb(naming::id_type const& contgid, naming::address&& addr,
        naming::id_type const& gid, threads::thread_priority priority,
        Callback && cb, Ts&&... vs)
    {
        typedef
            typename hpx::actions::extract_action<Action>::result_type
            result_type;

        return apply_p_cb<Action>(
            boost::make_shared<
                actions::typed_continuation<result_type>
            >(contgid),
            std::move(addr), gid, priority, std::forward<Callback>(cb),
            std::forward<Ts>(vs)...);
    }

    template <typename Action, typename Callback, typename ...Ts>
    inline bool
    apply_c_cb(naming::id_type const& contgid, naming::address&& addr,
        naming::id_type const& gid, Callback && cb, Ts&&... vs)
    {
        typedef
            typename hpx::actions::extract_action<Action>::result_type
            result_type;

        return apply_p_cb<Action>(
            boost::make_shared<
                actions::typed_continuation<result_type>
            >(contgid),
            std::move(addr), gid, actions::action_priority<Action>(),
            std::forward<Callback>(cb), std::forward<Ts>(vs)...);
    }
}

#endif
