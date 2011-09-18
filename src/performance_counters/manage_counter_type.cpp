////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#include <hpx/version.hpp>

#if HPX_AGAS_VERSION > 0x10

#include <hpx/runtime.hpp>
#include <hpx/performance_counters/manage_counter_type.hpp>

namespace hpx { namespace performance_counters
{
    void counter_type_shutdown(boost::shared_ptr<manage_counter_type> const& p)
    {
        BOOST_ASSERT(p); 
        p->uninstall();
    }

    void install_counter_type(std::string const& name,
        counter_type type, error_code& ec)
    {
        boost::shared_ptr<manage_counter_type> p(new manage_counter_type);

        // Install the counter type.
        p->install(name, type, ec);  

        // Register the shutdown function which will clean up this counter type.
        get_runtime().add_shutdown_function(
            boost::bind(&counter_type_shutdown, p));
    }

    void install_counter_types(counter_type_data const* data, std::size_t count, 
        error_code& ec)
    {
        for (std::size_t i = 0; i < count; ++i) 
        {
            install_counter_type(data[i].name_, data[i].type_, ec);
            if (ec) break;
        }
    } 
}}

#endif

