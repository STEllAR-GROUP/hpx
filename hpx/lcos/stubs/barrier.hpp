//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_LCOS_STUBS_BARRIER_MAR_10_2010_0306PM)
#define HPX_LCOS_STUBS_BARRIER_MAR_10_2010_0306PM

#include <hpx/runtime/components/stubs/stub_base.hpp>
#include <hpx/async.hpp>
#include <hpx/lcos/base_lco.hpp>
#include <hpx/lcos/server/barrier.hpp>

#include <boost/exception_ptr.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace lcos { namespace stubs
{
    struct barrier
      : public components::stub_base<lcos::server::barrier>
    {
        static lcos::future<void>
        wait_async(naming::id_type const& gid)
        {
            typedef lcos::base_lco::set_event_action action_type;
            return hpx::async<action_type>(gid);
        }

        static lcos::future<void>
        set_exception_async(naming::id_type const& gid,
            boost::exception_ptr const& e)
        {
            typedef lcos::base_lco::set_exception_action action_type;
            return hpx::async<action_type>(gid, e);
        }

        static void wait(naming::id_type const& gid)
        {
            wait_async(gid).get();
        }

        static void set_exception(naming::id_type const& gid,
            boost::exception_ptr const& e)
        {
            set_exception_async(gid, e).get();
        }
    };
}}}

#endif

