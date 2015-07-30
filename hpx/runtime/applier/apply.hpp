//  Copyright (c) 2007-2015 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_APPLIER_APPLY_NOV_27_2008_0957AM)
#define HPX_APPLIER_APPLY_NOV_27_2008_0957AM

#include <hpx/hpx_fwd.hpp>
#include <hpx/exception.hpp>

#include <hpx/runtime/agas/interface.hpp>
#include <hpx/runtime/parcelset/parcel.hpp>
#include <hpx/runtime/parcelset/parcelhandler.hpp>
#include <hpx/runtime/applier/applier.hpp>
#include <hpx/runtime/applier/apply_helper.hpp>
#include <hpx/runtime/applier/detail/apply_implementations.hpp>
#include <hpx/runtime/actions/component_action.hpp>
#include <hpx/runtime/actions/continuation.hpp>
#include <hpx/runtime/actions/transfer_action.hpp>
#include <hpx/traits/action_is_target_valid.hpp>
#include <hpx/traits/action_priority.hpp>
#include <hpx/traits/component_type_is_compatible.hpp>
#include <hpx/traits/is_action.hpp>
#include <hpx/traits/is_distribution_policy.hpp>

#if defined(HPX_SUPPORT_MULTIPLE_PARCEL_DESTINATIONS)
#include <hpx/util/remove_local_destinations.hpp>
#include <boost/dynamic_bitset.hpp>
#endif
#include <boost/format.hpp>
#include <boost/make_shared.hpp>

#include <vector>
#include <map>
#include <algorithm>
#include <type_traits>
#include <memory>

// FIXME: Error codes?

namespace hpx { namespace actions
{
    template <typename Action>
    threads::thread_priority action_priority()
    {
        typedef typename hpx::actions::extract_action<Action>::type action_type_;
        threads::thread_priority priority =
            static_cast<threads::thread_priority>(
                traits::action_priority<action_type_>::value);
        if (priority == threads::thread_priority_default)
            priority = threads::thread_priority_normal;
        return priority;
    }
}}

namespace hpx
{
    ///////////////////////////////////////////////////////////////////////////
    // zero parameter version of apply()
    // Invoked by a running HPX-thread to apply an action to any resource
    namespace applier { namespace detail
    {
        template <typename Action>
        inline naming::address& complement_addr(naming::address& addr)
        {
            if (components::component_invalid == addr.type_)
            {
                addr.type_ = components::get_component_type<
                    typename Action::component_type>();
            }
            return addr;
        }

        template <typename Action, typename ...Ts>
        inline bool
        put_parcel(naming::id_type const& id, naming::address&& addr,
            threads::thread_priority priority, Ts&&... vs)
        {
            typedef
                typename hpx::actions::extract_action<Action>::type
                action_type;

            std::unique_ptr<actions::base_action> action(
                new hpx::actions::transfer_action<action_type>(priority,
                    std::forward<Ts>(vs)...));
            parcelset::parcelhandler& ph =
                hpx::applier::get_applier().get_parcel_handler();

            parcelset::parcel p(id, complement_addr<action_type>(addr),
                std::move(action));

            ph.put_parcel(std::move(p));

            return false;     // destinations are remote
        }

        template <typename Action, typename ...Ts>
        inline bool
        put_parcel_cont(naming::id_type const& id, naming::address&& addr,
            threads::thread_priority priority,
            actions::continuation_type const& cont, Ts&&... vs)
        {
            typedef
                typename hpx::actions::extract_action<Action>::type
                action_type;

            std::unique_ptr<actions::base_action> action(
                new hpx::actions::transfer_action<action_type>(priority,
                    std::forward<Ts>(vs)...));
            parcelset::parcelhandler& ph =
                hpx::applier::get_applier().get_parcel_handler();

            parcelset::parcel p(id, complement_addr<action_type>(addr),
                std::move(action), cont);

            ph.put_parcel(std::move(p));

            return false;     // destinations are remote
        }

        template <typename Action, typename ...Ts>
        inline bool
        put_parcel_cb(naming::id_type const& id, naming::address&& addr,
            threads::thread_priority priority,
            parcelset::parcelhandler::write_handler_type const& cb, Ts&&... vs)
        {
            typedef
                typename hpx::actions::extract_action<Action>::type
                action_type;

            std::unique_ptr<actions::base_action> action(
                new hpx::actions::transfer_action<action_type>(priority,
                    std::forward<Ts>(vs)...));
            parcelset::parcelhandler& ph =
                hpx::applier::get_applier().get_parcel_handler();

            parcelset::parcel p(id, complement_addr<action_type>(addr),
                std::move(action));

            ph.put_parcel(std::move(p), cb);

            return false;     // destinations are remote
        }

        template <typename Action, typename ...Ts>
        inline bool
        put_parcel_cont_cb(naming::id_type const& id,
            naming::address&& addr, threads::thread_priority priority,
            actions::continuation_type const& cont,
            parcelset::parcelhandler::write_handler_type const& cb, Ts&&... vs)
        {
            typedef
                typename hpx::actions::extract_action<Action>::type
                action_type;

            std::unique_ptr<actions::base_action> action(
                new hpx::actions::transfer_action<action_type>(priority,
                    std::forward<Ts>(vs)...));
            parcelset::parcelhandler& ph =
                hpx::applier::get_applier().get_parcel_handler();

            parcelset::parcel p(id, complement_addr<action_type>(addr),
                std::move(action), cont);

            ph.put_parcel(std::move(p), cb);

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

#if defined(HPX_SUPPORT_MULTIPLE_PARCEL_DESTINATIONS)
        struct destinations
        {
            std::vector<naming::id_type> gids_;
            std::vector<naming::address> addrs_;
        };

        struct send_parcel
        {
            send_parcel(parcelset::parcelhandler& ph, actions::action_type act)
              : ph_(ph), act_(act)
            {}

            void operator()(
                std::pair<parcelset::locality const, destinations>& e) const
            {
                // Create a new parcel to be sent to the destination with the
                // gid, action, and arguments
                parcelset::parcel p(e.second.gids_, e.second.addrs_, act_);

                // Send the parcel through the parcel handler
                ph_.put_parcel(std::move(p));
            }

            parcelset::parcelhandler& ph_;
            actions::action_type act_;
        };

        template <typename Action, typename ...Ts>
        inline bool
        apply_r_p(std::vector<naming::address>& addrs,
            std::vector<naming::gid_type> const& gids,
            threads::thread_priority priority, Ts&&... vs)
        {
            typedef typename hpx::actions::extract_action<Action>::type action_type;

            // sort destinations
            std::map<parcelset::locality, destinations> dests;

            std::size_t count = gids.size();
            for (std::size_t i = 0; i < count; ++i) {
                complement_addr<action_type>(addrs[i]);

                destinations& dest = dests[addrs[i].locality_];
                dest.gids_.push_back(gids[i]);
                dest.addrs_.push_back(addrs[i]);
            }

            // send one parcel to each of the destination localities
            parcelset::parcelhandler& ph =
                hpx::applier::get_applier().get_parcel_handler();
            actions::action_type act(
                new hpx::actions::transfer_action<action_type>(priority,
                    std::forward<Ts>(vs)...));

            std::for_each(dests.begin(), dests.end(), send_parcel(ph, act));

            return false;     // destinations are remote
        }
#endif

        template <typename Action, typename ...Ts>
        inline bool
        apply_r (naming::address&& addr, naming::id_type const& gid,
            Ts&&... vs)
        {
            return apply_r_p<Action>(std::move(addr), gid,
                actions::action_priority<Action>(),
                std::forward<Ts>(vs)...);
        }

        // We know it is local and has to be directly executed.
        template <typename Action, typename ...Ts>
        inline bool
        apply_l_p(naming::id_type const& target, naming::address const& addr,
            threads::thread_priority priority, Ts&&... vs)
        {
            typedef typename hpx::actions::extract_action<Action>::type action_type;

            HPX_ASSERT(traits::component_type_is_compatible<
                typename action_type::component_type>::call(addr));

            apply_helper<action_type>::call(target, addr.address_, priority,
                std::forward<Ts>(vs)...);
            return true;     // no parcel has been sent (dest is local)
        }

        // same as above, but taking all arguments by value
        template <typename Action, typename ...Ts>
        inline bool
        apply_l_p_val(naming::id_type const& target, naming::address const& addr,
            threads::thread_priority priority, Ts... vs)
        {
            typedef typename hpx::actions::extract_action<Action>::type action_type;

            HPX_ASSERT(traits::component_type_is_compatible<
                typename action_type::component_type>::call(addr));

            apply_helper<action_type>::call(target, addr.address_, priority,
                std::move(vs)...);
            return true;     // no parcel has been sent (dest is local)
        }

        template <typename Action, typename ...Ts>
        inline bool
        apply_l (naming::id_type const& target, naming::address const& addr,
            Ts&&... vs)
        {
            return apply_l_p<Action>(target, addr,
                actions::action_priority<Action>(),
                std::forward<Ts>(vs)...);
        }
    }}

    ///////////////////////////////////////////////////////////////////////////
    template <typename Action, typename ...Ts>
    inline bool
    apply_p(naming::id_type const& gid, threads::thread_priority priority,
        Ts&&... vs)
    {
        return hpx::detail::apply_impl<Action>(
            actions::continuation_type(), gid, priority,
            std::forward<Ts>(vs)...);
    }

    template <typename Action, typename DistPolicy, typename ...Ts>
    inline typename boost::enable_if_c<
        traits::is_distribution_policy<DistPolicy>::value, bool
    >::type
    apply_p(DistPolicy const& policy, threads::thread_priority priority,
        Ts&&... vs)
    {
        return policy.template apply<Action>(
            actions::continuation_type(), priority, std::forward<Ts>(vs)...);
    }

    namespace detail
    {
        // dispatch point used for apply implementations
        template <typename Func, typename Enable = void>
        struct apply_dispatch;

        template <typename Action>
        struct apply_dispatch<Action,
            typename boost::enable_if_c<
                traits::is_action<Action>::value
            >::type>
        {
            template <typename Component, typename Signature, typename Derived,
                typename ...Ts>
            BOOST_FORCEINLINE static bool
            call(hpx::actions::basic_action<Component, Signature, Derived>,
                naming::id_type const& id, Ts&&... ts)
            {
                return apply_p<Derived>(id, actions::action_priority<Derived>(),
                    std::forward<Ts>(ts)...);
            }

            template <typename Component, typename Signature, typename Derived,
                typename DistPolicy, typename ...Ts>
            BOOST_FORCEINLINE static typename boost::enable_if_c<
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
    apply(naming::id_type const& gid, Ts&&... vs)
    {
        return apply_p<Action>(gid, actions::action_priority<Action>(),
            std::forward<Ts>(vs)...);
    }

    template <typename Action, typename DistPolicy, typename ...Ts>
    inline typename boost::enable_if_c<
        traits::is_distribution_policy<DistPolicy>::value, bool
    >::type
    apply(DistPolicy const& policy, Ts&&... vs)
    {
        return apply_p<Action>(policy, actions::action_priority<Action>(),
            std::forward<Ts>(vs)...);
    }

#if defined(HPX_SUPPORT_MULTIPLE_PARCEL_DESTINATIONS)
    // same for multiple destinations
    template <typename Action, typename ...Ts>
    inline bool
    apply_p(std::vector<naming::id_type> const& ids,
        threads::thread_priority priority, Ts&&... vs)
    {
        // Determine whether the gids are local or remote
        std::vector<naming::gid_type> gids;
        std::vector<naming::address> addrs;
        boost::dynamic_bitset<> locals;

        std::size_t count = ids.size();
        gids.reserve(count);
        if (agas::is_local_address_cached(ids, addrs, locals)) {
            // at least one destination is local
            for (std::size_t i = 0; i < count; ++i) {
                if (locals.test(i)) {
                    // apply locally, do not move arguments
                    applier::detail::apply_l_p_val<Action>(ids[i], addrs[i],
                        priority, vs...);
                }
                gids.push_back(applier::detail::convert_to_gid(ids[i]));
            }

            // remove local destinations
            std::vector<naming::gid_type>::iterator it =
                util::remove_local_destinations(gids, addrs, locals);
            if (it == gids.begin())
                return true;        // all destinations are local

            gids.erase(it, gids.end());
            addrs.resize(gids.size());
        }
        else {
            std::transform(ids.begin(), ids.end(), std::back_inserter(gids),
                applier::detail::convert_to_gid);
        }

        // apply remotely
        return applier::detail::apply_r_p<Action>(addrs, gids, priority,
            std::forward<Ts>(vs)...);
    }

    template <typename Action, typename ...Ts>
    inline bool
    apply (std::vector<naming::id_type> const& gids, Ts&&... vs)
    {
        return apply_p<Action>(gids, actions::action_priority<Action>(),
            std::forward<Ts>(vs)...);
    }

    template <typename Component, typename Signature, typename Derived,
        typename ...Ts>
    inline bool
    apply (
        hpx::actions::basic_action<Component, Signature, Derived> /*act*/,
        std::vector<naming::id_type> const& gids, Ts&&... vs)
    {
        return apply_p<Derived>(gids, actions::action_priority<Derived>(),
            std::forward<Ts>(vs)...);
    }
#endif

    ///////////////////////////////////////////////////////////////////////////
    namespace applier { namespace detail
    {
        template <typename Action, typename ...Ts>
        inline bool
        apply_r_p(naming::address&& addr, actions::continuation_type const& c,
            naming::id_type const& id, threads::thread_priority priority,
            Ts&&... vs)
        {
            if (!c)
            {
                return apply_r_p<Action>(std::move(addr), id, priority,
                    std::forward<Ts>(vs)...);
            }

            // If remote, create a new parcel to be sent to the destination
            // Create a new parcel with the gid, action, and arguments
            return detail::put_parcel_cont<Action>(id, std::move(addr),
                priority, c, std::forward<Ts>(vs)...);
        }

        template <typename Action, typename ...Ts>
        inline bool
        apply_r (naming::address&& addr, actions::continuation_type const& c,
            naming::id_type const& gid, Ts&&... vs)
        {
            return apply_r_p<Action>(std::move(addr), c, gid,
                actions::action_priority<Action>(),
                std::forward<Ts>(vs)...);
        }

        template <typename Action>
        inline bool
        apply_r_sync_p(naming::address&& addr, naming::id_type const& id,
            threads::thread_priority priority)
        {
            typedef typename hpx::actions::extract_action<Action>::type action_type_;

            // If remote, create a new parcel to be sent to the destination
            // Create a new parcel with the gid, action, and arguments
            std::unique_ptr<actions::base_action> act(
                new hpx::actions::transfer_action<action_type_>(priority));
            parcelset::parcel p(id, complement_addr<action_type_>(addr),
                std::move(act));

            // Send the parcel through the parcel handler
            hpx::applier::get_applier().get_parcel_handler().sync_put_parcel(std::move(p));
            return false;     // destination is remote
        }

        template <typename Action>
        inline bool
        apply_r_sync (naming::address&& addr, naming::id_type const& gid)
        {
            return apply_r_sync_p<Action>(std::move(addr), gid,
                actions::action_priority<Action>());
        }

        // We know it is local and has to be directly executed.
        template <typename Action, typename ...Ts>
        inline bool
        apply_l_p(actions::continuation_type const& cont,
            naming::id_type const& target, naming::address&& addr,
            threads::thread_priority priority, Ts&&... vs)
        {
            if (!cont)
            {
                return apply_l_p<Action>(target, std::move(addr), priority,
                    std::forward<Ts>(vs)...);
            }

            typedef typename hpx::actions::extract_action<
                    Action
                >::type action_type;

            HPX_ASSERT(traits::component_type_is_compatible<
                typename action_type::component_type>::call(addr));

            apply_helper<action_type>::call(
                cont, target, addr.address_, priority,
                std::forward<Ts>(vs)...);
            return true;     // no parcel has been sent (dest is local)
        }

        template <typename Action, typename ...Ts>
        inline bool
        apply_l (actions::continuation_type const& c, naming::id_type const& target,
            naming::address& addr, Ts&&... vs)
        {
            return apply_l_p<Action>(c, target, std::move(addr),
                actions::action_priority<Action>(), std::forward<Ts>(vs)...);
        }
    }}

    ///////////////////////////////////////////////////////////////////////////
    template <typename Action, typename ...Ts>
    inline bool
    apply_p(actions::continuation_type const& c, naming::id_type const& gid,
        threads::thread_priority priority, Ts&&... vs)
    {
        return hpx::detail::apply_impl<Action>(
            c, gid, priority, std::forward<Ts>(vs)...);
    }

    template <typename Action, typename DistPolicy, typename ...Ts>
    inline typename boost::enable_if_c<
        traits::is_distribution_policy<DistPolicy>::value, bool
    >::type
    apply_p(actions::continuation_type const& c, DistPolicy const& policy,
        threads::thread_priority priority, Ts&&... vs)
    {
        return policy.template apply<Action>(
            c, priority, std::forward<Ts>(vs)...);
    }

    namespace detail
    {
        template <>
        struct apply_dispatch<actions::continuation_type>
        {
            template <typename Component, typename Signature, typename Derived,
                typename ...Ts>
            BOOST_FORCEINLINE static bool
            call(actions::continuation_type const& c,
                hpx::actions::basic_action<Component, Signature, Derived>,
                naming::id_type const& id, Ts&&... ts)
            {
                return apply_p<Derived>(c, id,
                    actions::action_priority<Derived>(),
                    std::forward<Ts>(ts)...);
            }

            template <typename Component, typename Signature, typename Derived,
                typename DistPolicy, typename ...Ts>
            BOOST_FORCEINLINE static typename boost::enable_if_c<
                traits::is_distribution_policy<DistPolicy>::value, bool
            >::type
            call(actions::continuation_type const& c,
                hpx::actions::basic_action<Component, Signature, Derived>,
                DistPolicy const& policy, Ts&&... ts)
            {
                return apply_p<Derived>(c, policy,
                    actions::action_priority<Derived>(),
                    std::forward<Ts>(ts)...);
            }
        };
    }

    template <typename Action, typename ...Ts>
    inline bool
    apply(actions::continuation_type const& c, naming::id_type const& gid,
        Ts&&... vs)
    {
        return apply_p<Action>(c, gid, actions::action_priority<Action>(),
            std::forward<Ts>(vs)...);
    }

    template <typename Action, typename DistPolicy, typename ...Ts>
    inline typename boost::enable_if_c<
        traits::is_distribution_policy<DistPolicy>::value, bool
    >::type
    apply(actions::continuation_type const& c, DistPolicy const& policy,
        Ts&&... vs)
    {
        return apply_p<Action>(c, policy, actions::action_priority<Action>(),
            std::forward<Ts>(vs)...);
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
                typename hpx::actions::extract_action<Action>::result_type
                result_type;

            return apply_r_p<Action>(std::move(addr),
                boost::make_shared<
                    actions::typed_continuation<result_type>
                >(contgid),
                gid, priority, std::forward<Ts>(vs)...);
        }

        template <typename Action, typename ...Ts>
        inline bool
        apply_c (naming::address&& addr, naming::id_type const& contgid,
            naming::id_type const& gid, Ts&&... vs)
        {
            typedef
                typename hpx::actions::extract_action<Action>::result_type
                result_type;

            return apply_r_p<Action>(std::move(addr),
                boost::make_shared<
                    actions::typed_continuation<result_type>
                >(contgid),
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
            typename hpx::actions::extract_action<Action>::result_type
            result_type;

        return apply_p<Action>(
            boost::make_shared<
                actions::typed_continuation<result_type>
            >(contgid),
            gid, priority, std::forward<Ts>(vs)...);
    }

    template <typename Action, typename ...Ts>
    inline bool
    apply_c (naming::id_type const& contgid, naming::id_type const& gid,
        Ts&&... vs)
    {
        typedef
            typename hpx::actions::extract_action<Action>::result_type
            result_type;

        return apply_p<Action>(
            boost::make_shared<
                actions::typed_continuation<result_type>
            >(contgid),
            gid, actions::action_priority<Action>(),
            std::forward<Ts>(vs)...);
    }

    template <typename Component, typename Signature, typename Derived,
        typename ...Ts>
    inline bool
    apply_c (
        hpx::actions::basic_action<Component, Signature, Derived> /*act*/,
        naming::id_type const& contgid, naming::id_type const& gid,
        Ts&&... vs)
    {
        typedef
            typename hpx::actions::extract_action<Derived>::result_type
            result_type;

        return apply_p<Derived>(
            boost::make_shared<
                actions::typed_continuation<result_type>
            >(contgid),
            gid, actions::action_priority<Derived>(),
            std::forward<Ts>(vs)...);
    }
}

#endif
