//  Copyright (c) 2007-2011 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/applier/apply.hpp>
#include <hpx/runtime/actions/continuation.hpp>
#include <hpx/lcos/base_lco.hpp>

// #include <hpx/util/portable_binary_iarchive.hpp>
// #include <hpx/util/portable_binary_oarchive.hpp>
//
// #include <boost/serialization/version.hpp>
// #include <boost/serialization/export.hpp>

///////////////////////////////////////////////////////////////////////////////
// enable serialization of continuations through shared_ptr's
// BOOST_CLASS_EXPORT(hpx::actions::continuation);

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace actions
{
    ///////////////////////////////////////////////////////////////////////////
    void continuation::trigger()
    {
        LLCO_(info) << "promise::set(" << gid_ << ")";
        hpx::applier::apply<lcos::base_lco::set_event_action>(gid_);
    }

    ///////////////////////////////////////////////////////////////////////////
    void continuation::trigger_error(boost::exception_ptr const& e)
    {
        LLCO_(info) << "promise::set_error(" << gid_ << ")";
        hpx::applier::apply<lcos::base_lco::set_error_action>(gid_, e);
    }

    ///////////////////////////////////////////////////////////////////////////
    void continuation::enumerate_argument_gids(enum_gid_handler_type f)
    {
        f (boost::ref(gid_));
    }
}}

