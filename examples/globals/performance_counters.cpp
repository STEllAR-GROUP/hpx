///////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach 
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
///////////////////////////////////////////////////////////////////////////////

#include <hpx/hpx_init.hpp>
#include <hpx/include/iostreams.hpp>
#include <hpx/performance_counters/manage_counter_type.hpp>
#include <hpx/performance_counters/manage_counter.hpp>
#include <hpx/util/static.hpp>

using boost::program_options::variables_map;
using boost::program_options::options_description;

using hpx::cout;
using hpx::endl;

using hpx::init;
using hpx::finalize;

using hpx::performance_counters::manage_counter_type;
using hpx::performance_counters::manage_counter;
using hpx::performance_counters::counter_raw;

using hpx::util::static_;

struct perf_counter_foo_type {};
struct perf_counter_foo_bar {};

///////////////////////////////////////////////////////////////////////////////
boost::int64_t bar() { return 42; }

///////////////////////////////////////////////////////////////////////////////
void startup_()
{
    static_<manage_counter_type*, perf_counter_foo_type>
      type_(new manage_counter_type);
    static_<manage_counter*, perf_counter_foo_bar>
      instance_(new manage_counter);

    type_.get()->install("/foo/buzz", counter_raw);  
    instance_.get()->install("/foo(bar)/buzz", bar); 
}

///////////////////////////////////////////////////////////////////////////////
void shutdown_()
{
    static_<manage_counter_type*, perf_counter_foo_type> type_;
    static_<manage_counter*, perf_counter_foo_bar> instance_;

    delete type_.get();
    delete instance_.get();
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
       desc_commandline("usage: " HPX_APPLICATION_STRING " [options]");

    // Initialize and run HPX
    return init(desc_commandline, argc, argv, startup_, shutdown_);
}

