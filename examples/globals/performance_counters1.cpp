///////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach 
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
///////////////////////////////////////////////////////////////////////////////

#include <boost/bind.hpp>

#include <hpx/hpx_init.hpp>
#include <hpx/runtime.hpp>
#include <hpx/performance_counters/manage_counter_type.hpp>
#include <hpx/performance_counters/manage_counter.hpp>

using boost::program_options::variables_map;
using boost::program_options::options_description;

using boost::bind;

using hpx::get_runtime;

using hpx::init;
using hpx::finalize;

using hpx::performance_counters::manage_counter_type;
using hpx::performance_counters::manage_counter;
using hpx::performance_counters::counter_raw;

///////////////////////////////////////////////////////////////////////////////
boost::int64_t bar() { return 42; }

///////////////////////////////////////////////////////////////////////////////
void shutdown_(
    boost::shared_ptr<manage_counter_type> const& type_
  , boost::shared_ptr<manage_counter> const& instance_
) {
    BOOST_ASSERT(type_); 
    BOOST_ASSERT(instance_); 
    type_->uninstall();
    instance_->uninstall();
}

///////////////////////////////////////////////////////////////////////////////
void startup_()
{
    boost::shared_ptr<manage_counter_type> type_(new manage_counter_type);
    boost::shared_ptr<manage_counter> instance_(new manage_counter);

    // Install a counter type and a counter instance.
    type_->install("/foo/buzz", counter_raw);  
    instance_->install("/foo(bar)/buzz", bar); 

    // Register the shutdown function which will clean up the counter type and
    // instance.
    get_runtime().add_shutdown_function(bind(&shutdown_, type_, instance_));
}

///////////////////////////////////////////////////////////////////////////////
int hpx_main(variables_map& vm)
{
    // Do nothing.
    finalize();
    return 0;
}

///////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
    // Configure application-specific options
    options_description
       desc_commandline("Usage: " HPX_APPLICATION_STRING " [options]");
    
    // Initialize and run HPX
    return init(desc_commandline, argc, argv, startup_);
}

