//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/version.hpp>
#include <hpx/runtime.hpp>
#include <hpx/performance_counters/manage_counter.hpp>
#include <hpx/performance_counters/counter_creators.hpp>
#include <hpx/runtime/actions/continuation.hpp>

namespace hpx { namespace performance_counters
{
    counter_status manage_counter::install(naming::id_type const& id,
        counter_info const& info, error_code& ec)
    {
        if (0 != counter_) {
            HPX_THROWS_IF(ec, hpx::invalid_status, "manage_counter::install",
                "counter has been already installed");
            return status_invalid_data;
        }

        info_ = info;
        counter_ = id;

        return detail::add_counter(id, info_, ec);
    }

    void manage_counter::uninstall()
    {
        if (counter_)
        {
            error_code ec(lightweight);
            detail::remove_counter(info_, counter_, ec);
            counter_ = naming::invalid_id;
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    inline void counter_shutdown(boost::shared_ptr<manage_counter> const& p)
    {
        BOOST_ASSERT(p);
        p->uninstall();
    }

    void install_counter(naming::id_type const& id, counter_info const& info,
        error_code& ec)
    {
        boost::shared_ptr<manage_counter> p = boost::make_shared<manage_counter>();

        // Install the counter instance.
        p->install(id, info, ec);

        // Register the shutdown function which will clean up this counter.
        get_runtime().add_shutdown_function(boost::bind(&counter_shutdown, p));
    }
}}

