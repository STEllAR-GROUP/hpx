//  Copyright (c) 2007-2014 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#include <hpx/exception.hpp>
#include <hpx/apply.hpp>
#include <hpx/traits/action_priority.hpp>
#include <hpx/traits/extract_action.hpp>
#include <hpx/runtime/actions/continuation.hpp>
#include <hpx/lcos/base_lco.hpp>

#include <boost/exception_ptr.hpp>

#include <utility>

///////////////////////////////////////////////////////////////////////////////
namespace hpx
{
    void trigger_lco_event(naming::id_type const& id, naming::address && addr,
        bool move_credits)
    {
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
    }

    void trigger_lco_event(naming::id_type const& id, naming::address && addr,
        naming::id_type const& cont, bool move_credits)
    {
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
    }

    void set_lco_error(naming::id_type const& id, naming::address && addr,
        boost::exception_ptr const& e, bool move_credits)
    {
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
    }

    void set_lco_error(naming::id_type const& id, naming::address && addr, //-V659
        boost::exception_ptr && e, bool move_credits)
    {
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
    }

    void set_lco_error(naming::id_type const& id, naming::address && addr,
        boost::exception_ptr const& e, naming::id_type const& cont,
        bool move_credits)
    {
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
    }

    void set_lco_error(naming::id_type const& id, naming::address && addr, //-V659
        boost::exception_ptr && e, naming::id_type const& cont,
        bool move_credits)
    {
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

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace actions
{
    ///////////////////////////////////////////////////////////////////////////
    void continuation::trigger()
    {
        if (!gid_) {
            HPX_THROW_EXCEPTION(invalid_status,
                "continuation::trigger",
                "attempt to trigger invalid LCO (the id is invalid)");
            return;
        }

        LLCO_(info) << "continuation::trigger(" << gid_ << ")";
        trigger_lco_event(gid_, this->get_addr());
    }

    ///////////////////////////////////////////////////////////////////////////
    void continuation::trigger_error(boost::exception_ptr const& e)
    {
        if (!gid_) {
            HPX_THROW_EXCEPTION(invalid_status,
                "continuation::trigger_error",
                "attempt to trigger invalid LCO (the id is invalid)");
            return;
        }

        LLCO_(info) << "continuation::trigger_error(" << gid_ << ")";
        set_lco_error(gid_, this->get_addr(), e);
    }

    void continuation::trigger_error(boost::exception_ptr && e) //-V659
    {
        if (!gid_) {
            HPX_THROW_EXCEPTION(invalid_status,
                "continuation::trigger_error",
                "attempt to trigger invalid LCO (the id is invalid)");
            return;
        }

        LLCO_(info) << "continuation::trigger_error(" << gid_ << ")";
        set_lco_error(gid_, this->get_addr(), std::move(e));
    }
}}

HPX_REGISTER_TYPED_CONTINUATION(void, hpx_void_typed_continuation);

