////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#include <hpx/version.hpp>

#if HPX_AGAS_VERSION > 0x10

#include <hpx/runtime.hpp>
#include <hpx/performance_counters/manage_counter.hpp>

namespace hpx { namespace performance_counters
{
    inline void counter_shutdown(boost::shared_ptr<manage_counter> const& p)
    {
        BOOST_ASSERT(p);
        p->uninstall();
    }

    void install_counter(std::string const& name,
        boost::function<boost::int64_t()> const& f, error_code& ec)
    {
        boost::shared_ptr<manage_counter> p(new manage_counter);

        // Install the counter instance.
        p->install(name, f, ec);  

        // Register the shutdown function which will clean up this counter.
        get_runtime().add_shutdown_function(boost::bind(&counter_shutdown, p));
    }

    void install_counter(std::string const& name, error_code& ec)
    {
        boost::shared_ptr<manage_counter> p(new manage_counter);

        // Install the counter instance.
        p->install(name, ec);  

        // Register the shutdown function which will clean up this counter.
        get_runtime().add_shutdown_function(boost::bind(&counter_shutdown, p));
    }

    void install_counter(naming::id_type const& id, counter_info const& info, 
        error_code& ec)
    {
        boost::shared_ptr<manage_counter> p(new manage_counter);

        // Install the counter instance.
        p->install(id, info, ec);  

        // Register the shutdown function which will clean up this counter.
        get_runtime().add_shutdown_function(boost::bind(&counter_shutdown, p));
    }

    ///////////////////////////////////////////////////////////////////////////
    void install_counters(counter_data const* data, std::size_t count, 
        error_code& ec)
    {
        for (std::size_t i = 0; i < count; ++i) 
        {
            if (data[i].func_)
                install_counter(data[i].name_, data[i].func_, ec);
            else
                install_counter(data[i].name_, ec);
            if (ec) break;
        }
    } 
}}

#endif

