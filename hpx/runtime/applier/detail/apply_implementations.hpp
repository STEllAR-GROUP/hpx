//  Copyright (c) 2007-2015 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_APPLY_IMPLEMENTATIONS_APR_13_2015_0945AM)
#define HPX_APPLY_IMPLEMENTATIONS_APR_13_2015_0945AM

#include <hpx/config.hpp>
#include <hpx/runtime/applier/detail/apply_implementations_fwd.hpp>
#include <hpx/runtime/agas/interface.hpp>
#include <hpx/runtime/naming/address.hpp>
#include <hpx/runtime/naming/id_type.hpp>
#include <hpx/traits/is_continuation.hpp>
#include <hpx/util/move.hpp>

namespace hpx { namespace detail
{
    template <typename Action, typename Continuation, typename ...Ts>
    typename boost::enable_if_c<
        traits::is_continuation<Continuation>::value, bool
    >::type
    apply_impl(Continuation && c, hpx::id_type const& id,
        threads::thread_priority priority, Ts&&... vs)
    {
        if (!traits::action_is_target_valid<Action>::call(id)) {
            HPX_THROW_EXCEPTION(bad_parameter, "hpx::detail::apply_impl",
                boost::str(boost::format(
                    "the target (destination) does not match the action type (%s)"
                ) % hpx::actions::detail::get_action_name<Action>()));
            return false;
        }

        // Determine whether the id is local or remote
        naming::address addr;
        if (agas::is_local_address_cached(id, addr)) {
            return applier::detail::apply_l_p<Action>(std::forward<Continuation>(c),
                id, std::move(addr), priority, std::forward<Ts>(vs)...);
        }

        // apply remotely
        return applier::detail::apply_r_p<Action>(std::move(addr),
            std::forward<Continuation>(c), id, priority, std::forward<Ts>(vs)...);
    }

    template <typename Action, typename ...Ts>
    bool apply_impl(hpx::id_type const& id,
        threads::thread_priority priority, Ts&&... vs)
    {
        if (!traits::action_is_target_valid<Action>::call(id)) {
            HPX_THROW_EXCEPTION(bad_parameter, "hpx::detail::apply_impl",
                boost::str(boost::format(
                    "the target (destination) does not match the action type (%s)"
                ) % hpx::actions::detail::get_action_name<Action>()));
            return false;
        }

        // Determine whether the id is local or remote
        naming::address addr;
        if (agas::is_local_address_cached(id, addr)) {
            return applier::detail::apply_l_p<Action>(
                id, std::move(addr), priority, std::forward<Ts>(vs)...);
        }

        // apply remotely
        return applier::detail::apply_r_p<Action>(std::move(addr),
            id, priority, std::forward<Ts>(vs)...);
    }

    template <typename Action, typename Continuation, typename Callback, typename ...Ts>
    typename boost::enable_if_c<
        traits::is_continuation<Continuation>::value, bool
    >::type
    apply_cb_impl(Continuation && c, hpx::id_type const& id,
        threads::thread_priority priority, Callback&& cb, Ts&&... vs)
    {
        if (!traits::action_is_target_valid<Action>::call(id)) {
            HPX_THROW_EXCEPTION(bad_parameter, "hpx::detail::apply_cb_impl",
                boost::str(boost::format(
                    "the target (destination) does not match the action type (%s)"
                ) % hpx::actions::detail::get_action_name<Action>()));
            return false;
        }

        // Determine whether the gid is local or remote
        naming::address addr;
        if (agas::is_local_address_cached(id, addr)) {
            // apply locally
            bool result = applier::detail::apply_l_p<Action>(
                std::forward<Continuation>(c), id, std::move(addr), priority,
                std::forward<Ts>(vs)...);

            // invoke callback
            cb(boost::system::error_code(), parcelset::parcel());
            return result;
        }

        // apply remotely
        return applier::detail::apply_r_p_cb<Action>(std::move(addr),
            std::forward<Continuation>(c), id,
            priority, std::forward<Callback>(cb), std::forward<Ts>(vs)...);
    }

    template <typename Action, typename Callback, typename ...Ts>
    bool apply_cb_impl(hpx::id_type const& id,
        threads::thread_priority priority, Callback&& cb, Ts&&... vs)
    {
        if (!traits::action_is_target_valid<Action>::call(id)) {
            HPX_THROW_EXCEPTION(bad_parameter, "hpx::detail::apply_cb_impl",
                boost::str(boost::format(
                    "the target (destination) does not match the action type (%s)"
                ) % hpx::actions::detail::get_action_name<Action>()));
            return false;
        }

        // Determine whether the gid is local or remote
        naming::address addr;
        if (agas::is_local_address_cached(id, addr)) {
            // apply locally
            bool result = applier::detail::apply_l_p<Action>(
                id, std::move(addr), priority,
                std::forward<Ts>(vs)...);

            // invoke callback
            cb(boost::system::error_code(), parcelset::parcel());
            return result;
        }

        // apply remotely
        return applier::detail::apply_r_p_cb<Action>(std::move(addr), id,
            priority, std::forward<Callback>(cb), std::forward<Ts>(vs)...);
    }
}}

#endif
