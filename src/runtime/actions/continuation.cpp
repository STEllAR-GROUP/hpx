//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_fwd.hpp>
#include <hpx/exception.hpp>
#include <hpx/runtime/applier/apply.hpp>
#include <hpx/runtime/actions/continuation.hpp>
#include <hpx/lcos/base_lco.hpp>

BOOST_CLASS_EXPORT(hpx::actions::continuation)

///////////////////////////////////////////////////////////////////////////////
namespace hpx
{
    void trigger_lco_event(naming::id_type const& id)
    {
        lcos::base_lco::set_event_action set;
        apply(set, id);
    }

    void set_lco_error(naming::id_type const& id, boost::exception_ptr const& e)
    {
        lcos::base_lco::set_exception_action set;
        apply(set, id, e);
    }

    void set_lco_error(naming::id_type const& id,
        BOOST_RV_REF(boost::exception_ptr) e)
    {
        lcos::base_lco::set_exception_action set;
        apply(set, id, boost::move(e));
    }
}

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace actions
{
    ///////////////////////////////////////////////////////////////////////////
    void continuation::trigger() const
    {
        if (!gid_) {
            HPX_THROW_EXCEPTION(invalid_status,
                "continuation::trigger",
                "attempt to trigger invalid LCO (the id is invalid)");
            return;
        }

        LLCO_(info) << "continuation::trigger(" << gid_ << ")";
        trigger_lco_event(gid_);
    }

    ///////////////////////////////////////////////////////////////////////////
    void continuation::trigger_error(boost::exception_ptr const& e) const
    {
        if (!gid_) {
            HPX_THROW_EXCEPTION(invalid_status,
                "continuation::trigger_error",
                "attempt to trigger invalid LCO (the id is invalid)");
            return;
        }

        LLCO_(info) << "continuation::trigger_error(" << gid_ << ")";
        set_lco_error(gid_, e);
    }

    void continuation::trigger_error(BOOST_RV_REF(boost::exception_ptr) e) const
    {
        if (!gid_) {
            HPX_THROW_EXCEPTION(invalid_status,
                "continuation::trigger_error",
                "attempt to trigger invalid LCO (the id is invalid)");
            return;
        }

        LLCO_(info) << "continuation::trigger_error(" << gid_ << ")";
        set_lco_error(gid_, boost::move(e));
    }
}}

