//  Copyright (c) 2007-2022 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/actions_base/traits/action_is_target_valid.hpp>
#include <hpx/actions_base/traits/action_was_object_migrated.hpp>
#include <hpx/actions_base/traits/is_continuation.hpp>
#include <hpx/async_distributed/detail/post_implementations_fwd.hpp>
#include <hpx/components_base/agas_interface.hpp>
#include <hpx/components_base/traits/component_supports_migration.hpp>
#include <hpx/modules/errors.hpp>
#include <hpx/naming_base/address.hpp>
#include <hpx/naming_base/id_type.hpp>
#include <hpx/parcelset/parcel.hpp>

#include <system_error>
#include <type_traits>
#include <utility>

namespace hpx { namespace detail {
    template <typename Action, typename Continuation, typename... Ts>
    std::enable_if_t<traits::is_continuation<Continuation>::value, bool>
    post_impl(Continuation&& c, hpx::id_type const& id,
        threads::thread_priority priority, Ts&&... vs)
    {
        if (!traits::action_is_target_valid<Action>::call(id))
        {
            HPX_THROW_EXCEPTION(hpx::error::bad_parameter,
                "hpx::detail::post_impl",
                "the target (destination) does not match the action type ({})",
                hpx::actions::detail::get_action_name<Action>());
            return false;
        }

        std::pair<bool, components::pinned_ptr> r;

        // Determine whether the id is local or remote
        naming::address addr;
        if (agas::is_local_address_cached(id, addr))
        {
            using component_type = typename Action::component_type;
            if (traits::component_supports_migration<component_type>::call())
            {
                r = traits::action_was_object_migrated<Action>::call(
                    id, addr.address_);
                if (!r.first)
                {
                    return hpx::detail::post_l_p<Action>(
                        HPX_FORWARD(Continuation, c), id, HPX_MOVE(addr),
                        priority, HPX_FORWARD(Ts, vs)...);
                }
            }
            else
            {
                return hpx::detail::post_l_p<Action>(
                    HPX_FORWARD(Continuation, c), id, HPX_MOVE(addr), priority,
                    HPX_FORWARD(Ts, vs)...);
            }
        }

#if defined(HPX_HAVE_NETWORKING)
        // apply remotely
        return hpx::detail::post_r_p<Action>(HPX_MOVE(addr),
            HPX_FORWARD(Continuation, c), id, priority, HPX_FORWARD(Ts, vs)...);
#else
        HPX_THROW_EXCEPTION(hpx::error::invalid_status, "hpx::post_impl",
            "unexpected attempt to send a parcel with networking disabled");
#endif
    }

    template <typename Action, typename Continuation, typename... Ts>
    std::enable_if_t<traits::is_continuation<Continuation>::value, bool>
    post_impl(Continuation&& c, hpx::id_type const& id, naming::address&& addr,
        threads::thread_priority priority, Ts&&... vs)
    {
        // Determine whether the id is local or remote
        if (addr)
        {
            if (!traits::action_is_target_valid<Action>::call(id))
            {
                HPX_THROW_EXCEPTION(hpx::error::bad_parameter,
                    "hpx::detail::post_impl",
                    "the target (destination) does not match the action type "
                    "({})",
                    hpx::actions::detail::get_action_name<Action>());
                return false;
            }

            std::pair<bool, components::pinned_ptr> r;
            if (naming::get_locality_id_from_gid(addr.locality_) ==
                agas::get_locality_id())
            {
                using component_type = typename Action::component_type;
                if (traits::component_supports_migration<
                        component_type>::call())
                {
                    r = traits::action_was_object_migrated<Action>::call(
                        id, addr.address_);
                    if (!r.first)
                    {
                        return hpx::detail::post_l_p<Action>(
                            HPX_FORWARD(Continuation, c), id, HPX_MOVE(addr),
                            priority, HPX_FORWARD(Ts, vs)...);
                    }
                }
                else
                {
                    return hpx::detail::post_l_p<Action>(
                        HPX_FORWARD(Continuation, c), id, HPX_MOVE(addr),
                        priority, HPX_FORWARD(Ts, vs)...);
                }
            }
            // object was migrated or is not local
            else
            {
                // apply remotely
#if defined(HPX_HAVE_NETWORKING)
                return hpx::detail::post_r_p<Action>(HPX_MOVE(addr),
                    HPX_FORWARD(Continuation, c), id, priority,
                    HPX_FORWARD(Ts, vs)...);
#else
                HPX_THROW_EXCEPTION(hpx::error::invalid_status,
                    "hpx::detail::post_impl",
                    "unexpected attempt to send a parcel with networking "
                    "disabled");
#endif
            }
        }

        return post_impl<Action>(
            HPX_FORWARD(Continuation, c), id, priority, HPX_FORWARD(Ts, vs)...);
    }

    template <typename Action, typename... Ts>
    bool post_impl(
        hpx::id_type const& id, threads::thread_priority priority, Ts&&... vs)
    {
        if (!traits::action_is_target_valid<Action>::call(id))
        {
            HPX_THROW_EXCEPTION(hpx::error::bad_parameter,
                "hpx::detail::post_impl",
                "the target (destination) does not match the action type ({})",
                hpx::actions::detail::get_action_name<Action>());
            return false;
        }

        std::pair<bool, components::pinned_ptr> r;

        // Determine whether the id is local or remote
        naming::address addr;
        if (agas::is_local_address_cached(id, addr))
        {
            using component_type = typename Action::component_type;
            if (traits::component_supports_migration<component_type>::call())
            {
                r = traits::action_was_object_migrated<Action>::call(
                    id, addr.address_);
                if (!r.first)
                {
                    return hpx::detail::post_l_p<Action>(
                        id, HPX_MOVE(addr), priority, HPX_FORWARD(Ts, vs)...);
                }
            }
            else
            {
                return hpx::detail::post_l_p<Action>(
                    id, HPX_MOVE(addr), priority, HPX_FORWARD(Ts, vs)...);
            }
        }

#if defined(HPX_HAVE_NETWORKING)
        // apply remotely
        return hpx::detail::post_r_p<Action>(
            HPX_MOVE(addr), id, priority, HPX_FORWARD(Ts, vs)...);
#else
        HPX_THROW_EXCEPTION(hpx::error::invalid_status, "hpx::post_impl",
            "unexpected attempt to send a parcel with networking disabled");
#endif
    }

    template <typename Action, typename... Ts>
    bool post_impl(hpx::id_type const& id, naming::address&& addr,
        threads::thread_priority priority, Ts&&... vs)
    {
        // Determine whether the id is local or remote
        if (addr)
        {
            if (!traits::action_is_target_valid<Action>::call(id))
            {
                HPX_THROW_EXCEPTION(hpx::error::bad_parameter,
                    "hpx::detail::post_impl",
                    "the target (destination) does not match the action type "
                    "({})",
                    hpx::actions::detail::get_action_name<Action>());
                return false;
            }

            std::pair<bool, components::pinned_ptr> r;
            if (naming::get_locality_id_from_gid(addr.locality_) ==
                agas::get_locality_id())
            {
                using component_type = typename Action::component_type;
                if (traits::component_supports_migration<
                        component_type>::call())
                {
                    r = traits::action_was_object_migrated<Action>::call(
                        id, addr.address_);
                    if (!r.first)
                    {
                        return hpx::detail::post_l_p<Action>(id, HPX_MOVE(addr),
                            priority, HPX_FORWARD(Ts, vs)...);
                    }
                }
                else
                {
                    return hpx::detail::post_l_p<Action>(
                        id, HPX_MOVE(addr), priority, HPX_FORWARD(Ts, vs)...);
                }
            }
            // object was migrated or is not local
            else
            {
                // apply remotely
#if defined(HPX_HAVE_NETWORKING)
                return hpx::detail::post_r_p<Action>(
                    HPX_MOVE(addr), id, priority, HPX_FORWARD(Ts, vs)...);
#else
                HPX_THROW_EXCEPTION(hpx::error::invalid_status,
                    "hpx::detail::post_impl",
                    "unexpected attempt to send a parcel with networking "
                    "disabled");
#endif
            }
        }
        return post_impl<Action>(id, priority, HPX_FORWARD(Ts, vs)...);
    }

    template <typename Action, typename Continuation, typename Callback,
        typename... Ts>
    std::enable_if_t<traits::is_continuation<Continuation>::value, bool>
    post_cb_impl(Continuation&& c, hpx::id_type const& id,
        threads::thread_priority priority, Callback&& cb, Ts&&... vs)
    {
        if (!traits::action_is_target_valid<Action>::call(id))
        {
            HPX_THROW_EXCEPTION(hpx::error::bad_parameter,
                "hpx::detail::post_cb_impl",
                "the target (destination) does not match the action type ({})",
                hpx::actions::detail::get_action_name<Action>());
            return false;
        }

        std::pair<bool, components::pinned_ptr> r;

        // Determine whether the id is local or remote
        naming::address addr;
        if (agas::is_local_address_cached(id, addr))
        {
            using component_type = typename Action::component_type;
            if (traits::component_supports_migration<component_type>::call())
            {
                r = traits::action_was_object_migrated<Action>::call(
                    id, addr.address_);
                if (!r.first)
                {
                    bool result = hpx::detail::post_l_p<Action>(
                        HPX_FORWARD(Continuation, c), id, HPX_MOVE(addr),
                        priority, HPX_FORWARD(Ts, vs)...);

                    // invoke callback
#if defined(HPX_HAVE_NETWORKING)
                    cb(std::error_code(), parcelset::parcel());
#else
                    cb();
#endif
                    return result;
                }
            }
            else
            {
                bool result =
                    hpx::detail::post_l_p<Action>(HPX_FORWARD(Continuation, c),
                        id, HPX_MOVE(addr), priority, HPX_FORWARD(Ts, vs)...);

                // invoke callback
#if defined(HPX_HAVE_NETWORKING)
                cb(std::error_code(), parcelset::parcel());
#else
                cb();
#endif
                return result;
            }
        }

#if defined(HPX_HAVE_NETWORKING)
        // apply remotely
        return hpx::detail::post_r_p_cb<Action>(HPX_MOVE(addr),
            HPX_FORWARD(Continuation, c), id, priority,
            HPX_FORWARD(Callback, cb), HPX_FORWARD(Ts, vs)...);
#else
        HPX_THROW_EXCEPTION(hpx::error::invalid_status,
            "hpx::detail::post_cb_impl",
            "unexpected attempt to send a parcel with networking disabled");
#endif
    }

    template <typename Action, typename Callback, typename... Ts>
    bool post_cb_impl(hpx::id_type const& id, threads::thread_priority priority,
        Callback&& cb, Ts&&... vs)
    {
        if (!traits::action_is_target_valid<Action>::call(id))
        {
            HPX_THROW_EXCEPTION(hpx::error::bad_parameter,
                "hpx::detail::post_cb_impl",
                "the target (destination) does not match the action type ({})",
                hpx::actions::detail::get_action_name<Action>());
            return false;
        }

        std::pair<bool, components::pinned_ptr> r;

        // Determine whether the id is local or remote
        naming::address addr;
        if (agas::is_local_address_cached(id, addr))
        {
            using component_type = typename Action::component_type;
            if (traits::component_supports_migration<component_type>::call())
            {
                r = traits::action_was_object_migrated<Action>::call(
                    id, addr.address_);
                if (!r.first)
                {
                    bool result = hpx::detail::post_l_p<Action>(
                        id, HPX_MOVE(addr), priority, HPX_FORWARD(Ts, vs)...);

                    // invoke callback
#if defined(HPX_HAVE_NETWORKING)
                    cb(std::error_code(), parcelset::parcel());
#else
                    cb();
#endif
                    return result;
                }
            }
            else
            {
                bool result = hpx::detail::post_l_p<Action>(
                    id, HPX_MOVE(addr), priority, HPX_FORWARD(Ts, vs)...);

                // invoke callback
#if defined(HPX_HAVE_NETWORKING)
                cb(std::error_code(), parcelset::parcel());
#else
                cb();
#endif
                return result;
            }
        }

#if defined(HPX_HAVE_NETWORKING)
        // apply remotely
        return hpx::detail::post_r_p_cb<Action>(HPX_MOVE(addr), id, priority,
            HPX_FORWARD(Callback, cb), HPX_FORWARD(Ts, vs)...);
#else
        HPX_THROW_EXCEPTION(hpx::error::invalid_status,
            "hpx::detail::post_cb_impl",
            "unexpected attempt to send a parcel with networking disabled");
#endif
    }
}}    // namespace hpx::detail
