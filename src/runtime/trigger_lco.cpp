//  Copyright (c) 2007-2014 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/assert.hpp>
#include <hpx/async_distributed/apply.hpp>
#include <hpx/lcos/base_lco.hpp>
#include <hpx/naming_base/address.hpp>
#include <hpx/naming_base/id_type.hpp>
#include <hpx/runtime/trigger_lco.hpp>
#include <hpx/type_support/unused.hpp>
#if defined(HPX_MSVC) && !defined(HPX_DEBUG)
#include <hpx/lcos/base_lco_with_value.hpp>
#endif

#include <exception>
#include <utility>

namespace hpx
{
    void trigger_lco_event(naming::id_type const& id, naming::address&& addr,
        bool move_credits)
    {
#if !defined(HPX_COMPUTE_DEVICE_CODE)
        typedef lcos::base_lco::set_event_action set_action;
        if (move_credits &&
            id.get_management_type() != naming::id_type::unmanaged)
        {
            naming::id_type target(id.get_gid(), id_type::managed_move_credit);
            id.make_unmanaged();

            detail::apply_impl<set_action>(
                target, std::move(addr), actions::action_priority<set_action>());
        }
        else
        {
            detail::apply_impl<set_action>(
                id, std::move(addr), actions::action_priority<set_action>());
        }
#else
        HPX_ASSERT(false);
        HPX_UNUSED(id);
        HPX_UNUSED(addr);
        HPX_UNUSED(move_credits);
#endif
    }

    void trigger_lco_event(naming::id_type const& id, naming::address && addr,
        naming::id_type const& cont, bool move_credits)
    {
#if !defined(HPX_COMPUTE_DEVICE_CODE)
        typedef lcos::base_lco::set_event_action set_action;
        typedef
            hpx::traits::extract_action<set_action>::local_result_type
            local_result_type;
        typedef
            hpx::traits::extract_action<set_action>::remote_result_type
            remote_result_type;
        if (move_credits &&
            id.get_management_type() != naming::id_type::unmanaged)
        {
            naming::id_type target(id.get_gid(), id_type::managed_move_credit);
            id.make_unmanaged();

            detail::apply_impl<set_action>(
                actions::typed_continuation<
                    local_result_type, remote_result_type>(cont),
                target, std::move(addr), actions::action_priority<set_action>());
        }
        else
        {
            detail::apply_impl<set_action>(
                actions::typed_continuation<
                    local_result_type, remote_result_type>(cont),
                id, std::move(addr), actions::action_priority<set_action>());
        }
#else
        HPX_ASSERT(false);
        HPX_UNUSED(id);
        HPX_UNUSED(cont);
        HPX_UNUSED(addr);
        HPX_UNUSED(move_credits);
#endif
    }

    void set_lco_error(naming::id_type const& id, naming::address && addr,
        std::exception_ptr const& e, bool move_credits)
    {
#if !defined(HPX_COMPUTE_DEVICE_CODE)
        typedef lcos::base_lco::set_exception_action set_action;
        if (move_credits &&
            id.get_management_type() != naming::id_type::unmanaged)
        {
            naming::id_type target(id.get_gid(), id_type::managed_move_credit);
            id.make_unmanaged();

            detail::apply_impl<set_action>(
                target, std::move(addr), actions::action_priority<set_action>(), e);
        }
        else
        {
            detail::apply_impl<set_action>(
                id, std::move(addr), actions::action_priority<set_action>(), e);
        }
#else
        HPX_ASSERT(false);
        HPX_UNUSED(id);
        HPX_UNUSED(addr);
        HPX_UNUSED(e);
        HPX_UNUSED(move_credits);
#endif
    }

    void set_lco_error(naming::id_type const& id, naming::address && addr, //-V659
        std::exception_ptr && e, bool move_credits)
    {
#if !defined(HPX_COMPUTE_DEVICE_CODE)
        typedef lcos::base_lco::set_exception_action set_action;
        if (move_credits &&
            id.get_management_type() != naming::id_type::unmanaged)
        {
            naming::id_type target(id.get_gid(), id_type::managed_move_credit);
            id.make_unmanaged();

            detail::apply_impl<set_action>(
                target, std::move(addr), actions::action_priority<set_action>(),
                std::move(e));
        }
        else
        {
            detail::apply_impl<set_action>(
                id, std::move(addr), actions::action_priority<set_action>(),
                std::move(e));
        }
#else
        HPX_ASSERT(false);
        HPX_UNUSED(id);
        HPX_UNUSED(addr);
        HPX_UNUSED(e);
        HPX_UNUSED(move_credits);
#endif
    }

    void set_lco_error(naming::id_type const& id, naming::address && addr,
        std::exception_ptr const& e, naming::id_type const& cont,
        bool move_credits)
    {
#if !defined(HPX_COMPUTE_DEVICE_CODE)
        typedef lcos::base_lco::set_exception_action set_action;
        typedef
            hpx::traits::extract_action<set_action>::local_result_type
            local_result_type;
        typedef
            hpx::traits::extract_action<set_action>::remote_result_type
            remote_result_type;
        if (move_credits &&
            id.get_management_type() != naming::id_type::unmanaged)
        {
            naming::id_type target(id.get_gid(), id_type::managed_move_credit);
            id.make_unmanaged();

            detail::apply_impl<set_action>(
                actions::typed_continuation<
                    local_result_type, remote_result_type>(cont),
                target, std::move(addr), actions::action_priority<set_action>(), e);
        }
        else
        {
            detail::apply_impl<set_action>(
                actions::typed_continuation<
                    local_result_type, remote_result_type>(cont),
                id, std::move(addr), actions::action_priority<set_action>(), e);
        }
#else
        HPX_ASSERT(false);
        HPX_UNUSED(id);
        HPX_UNUSED(addr);
        HPX_UNUSED(e);
        HPX_UNUSED(cont);
        HPX_UNUSED(move_credits);
#endif
    }

    void set_lco_error(naming::id_type const& id, naming::address && addr, //-V659
        std::exception_ptr && e, naming::id_type const& cont,
        bool move_credits)
    {
#if !defined(HPX_COMPUTE_DEVICE_CODE)
        typedef lcos::base_lco::set_exception_action set_action;
        typedef
            hpx::traits::extract_action<set_action>::local_result_type
            local_result_type;
        typedef
            hpx::traits::extract_action<set_action>::remote_result_type
            remote_result_type;
        if (move_credits &&
            id.get_management_type() != naming::id_type::unmanaged)
        {
            naming::id_type target(id.get_gid(), id_type::managed_move_credit);
            id.make_unmanaged();

            detail::apply_impl<set_action>(
                actions::typed_continuation<
                    local_result_type, remote_result_type>(cont),
                target, std::move(addr),
                actions::action_priority<set_action>(), std::move(e));
        }
        else
        {
            detail::apply_impl<set_action>(
                actions::typed_continuation<
                    local_result_type, remote_result_type>(cont),
                id, std::move(addr),
                actions::action_priority<set_action>(), std::move(e));
        }
#else
        HPX_ASSERT(false);
        HPX_UNUSED(id);
        HPX_UNUSED(addr);
        HPX_UNUSED(e);
        HPX_UNUSED(cont);
        HPX_UNUSED(move_credits);
#endif
    }

#if defined(HPX_MSVC) && !defined(HPX_DEBUG)
    ///////////////////////////////////////////////////////////////////////////
    // Explicitly instantiate specific apply needed for set_lco_value for MSVC
    // (in release mode only, leads to missing symbols otherwise).
    template bool apply<
        lcos::base_lco_with_value<
            util::unused_type, util::unused_type
        >::set_value_action,
        util::unused_type
    >(naming::id_type const &, util::unused_type &&);
#endif
}

