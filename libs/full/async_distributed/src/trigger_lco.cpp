//  Copyright (c) 2007-2021 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/assert.hpp>
#include <hpx/async_distributed/base_lco.hpp>
#include <hpx/async_distributed/post.hpp>
#include <hpx/async_distributed/trigger_lco.hpp>

#include <hpx/naming_base/address.hpp>
#include <hpx/naming_base/id_type.hpp>
#include <hpx/type_support/unused.hpp>
#if defined(HPX_MSVC) && !defined(HPX_DEBUG)
#include <hpx/async_distributed/base_lco_with_value.hpp>
#endif

#include <exception>
#include <utility>

namespace hpx {

    void trigger_lco_event(
        hpx::id_type const& id, naming::address&& addr, bool move_credits)
    {
#if !defined(HPX_COMPUTE_DEVICE_CODE)
        typedef lcos::base_lco::set_event_action set_action;
        if (move_credits &&
            id.get_management_type() !=
                hpx::id_type::management_type::unmanaged)
        {
            hpx::id_type target(
                id.get_gid(), id_type::management_type::managed_move_credit);
            id.make_unmanaged();

            detail::post_impl<set_action>(
                target, HPX_MOVE(addr), actions::action_priority<set_action>());
        }
        else
        {
            detail::post_impl<set_action>(
                id, HPX_MOVE(addr), actions::action_priority<set_action>());
        }
#else
        HPX_ASSERT(false);
        HPX_UNUSED(id);
        HPX_UNUSED(addr);
        HPX_UNUSED(move_credits);
#endif
    }

    void trigger_lco_event(hpx::id_type const& id, naming::address&& addr,
        hpx::id_type const& cont, bool move_credits)
    {
#if !defined(HPX_COMPUTE_DEVICE_CODE)
        typedef lcos::base_lco::set_event_action set_action;
        typedef hpx::traits::extract_action<set_action>::local_result_type
            local_result_type;
        typedef hpx::traits::extract_action<set_action>::remote_result_type
            remote_result_type;
        if (move_credits &&
            id.get_management_type() !=
                hpx::id_type::management_type::unmanaged)
        {
            hpx::id_type target(
                id.get_gid(), id_type::management_type::managed_move_credit);
            id.make_unmanaged();

            detail::post_impl<set_action>(
                actions::typed_continuation<local_result_type,
                    remote_result_type>(cont),
                target, HPX_MOVE(addr), actions::action_priority<set_action>());
        }
        else
        {
            detail::post_impl<set_action>(
                actions::typed_continuation<local_result_type,
                    remote_result_type>(cont),
                id, HPX_MOVE(addr), actions::action_priority<set_action>());
        }
#else
        HPX_ASSERT(false);
        HPX_UNUSED(id);
        HPX_UNUSED(cont);
        HPX_UNUSED(addr);
        HPX_UNUSED(move_credits);
#endif
    }

    void set_lco_error(hpx::id_type const& id, naming::address&& addr,
        std::exception_ptr const& e, bool move_credits)
    {
#if !defined(HPX_COMPUTE_DEVICE_CODE)
        typedef lcos::base_lco::set_exception_action set_action;
        if (move_credits &&
            id.get_management_type() !=
                hpx::id_type::management_type::unmanaged)
        {
            hpx::id_type target(
                id.get_gid(), id_type::management_type::managed_move_credit);
            id.make_unmanaged();

            detail::post_impl<set_action>(target, HPX_MOVE(addr),
                actions::action_priority<set_action>(), e);
        }
        else
        {
            detail::post_impl<set_action>(
                id, HPX_MOVE(addr), actions::action_priority<set_action>(), e);
        }
#else
        HPX_ASSERT(false);
        HPX_UNUSED(id);
        HPX_UNUSED(addr);
        HPX_UNUSED(e);
        HPX_UNUSED(move_credits);
#endif
    }

    void set_lco_error(hpx::id_type const& id,
        naming::address&& addr,    //-V659
        std::exception_ptr&& e, bool move_credits)
    {
#if !defined(HPX_COMPUTE_DEVICE_CODE)
        typedef lcos::base_lco::set_exception_action set_action;
        if (move_credits &&
            id.get_management_type() !=
                hpx::id_type::management_type::unmanaged)
        {
            hpx::id_type target(
                id.get_gid(), id_type::management_type::managed_move_credit);
            id.make_unmanaged();

            detail::post_impl<set_action>(target, HPX_MOVE(addr),
                actions::action_priority<set_action>(), HPX_MOVE(e));
        }
        else
        {
            detail::post_impl<set_action>(id, HPX_MOVE(addr),
                actions::action_priority<set_action>(), HPX_MOVE(e));
        }
#else
        HPX_ASSERT(false);
        HPX_UNUSED(id);
        HPX_UNUSED(addr);
        HPX_UNUSED(e);
        HPX_UNUSED(move_credits);
#endif
    }

    void set_lco_error(hpx::id_type const& id, naming::address&& addr,
        std::exception_ptr const& e, hpx::id_type const& cont,
        bool move_credits)
    {
#if !defined(HPX_COMPUTE_DEVICE_CODE)
        typedef lcos::base_lco::set_exception_action set_action;
        typedef hpx::traits::extract_action<set_action>::local_result_type
            local_result_type;
        typedef hpx::traits::extract_action<set_action>::remote_result_type
            remote_result_type;
        if (move_credits &&
            id.get_management_type() !=
                hpx::id_type::management_type::unmanaged)
        {
            hpx::id_type target(
                id.get_gid(), id_type::management_type::managed_move_credit);
            id.make_unmanaged();

            detail::post_impl<set_action>(
                actions::typed_continuation<local_result_type,
                    remote_result_type>(cont),
                target, HPX_MOVE(addr), actions::action_priority<set_action>(),
                e);
        }
        else
        {
            detail::post_impl<set_action>(
                actions::typed_continuation<local_result_type,
                    remote_result_type>(cont),
                id, HPX_MOVE(addr), actions::action_priority<set_action>(), e);
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

    void set_lco_error(hpx::id_type const& id,
        naming::address&& addr,    //-V659
        std::exception_ptr&& e, hpx::id_type const& cont, bool move_credits)
    {
#if !defined(HPX_COMPUTE_DEVICE_CODE)
        typedef lcos::base_lco::set_exception_action set_action;
        typedef hpx::traits::extract_action<set_action>::local_result_type
            local_result_type;
        typedef hpx::traits::extract_action<set_action>::remote_result_type
            remote_result_type;
        if (move_credits &&
            id.get_management_type() !=
                hpx::id_type::management_type::unmanaged)
        {
            hpx::id_type target(
                id.get_gid(), id_type::management_type::managed_move_credit);
            id.make_unmanaged();

            detail::post_impl<set_action>(
                actions::typed_continuation<local_result_type,
                    remote_result_type>(cont),
                target, HPX_MOVE(addr), actions::action_priority<set_action>(),
                HPX_MOVE(e));
        }
        else
        {
            detail::post_impl<set_action>(
                actions::typed_continuation<local_result_type,
                    remote_result_type>(cont),
                id, HPX_MOVE(addr), actions::action_priority<set_action>(),
                HPX_MOVE(e));
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
    // Explicitly instantiate specific post needed for set_lco_value for MSVC
    // (in release mode only, leads to missing symbols otherwise).
    template bool post<lcos::base_lco_with_value<util::unused_type,
                           util::unused_type>::set_value_action,
        util::unused_type>(hpx::id_type const&, util::unused_type&&);
#endif
}    // namespace hpx
