//  Copyright (c) 2007-2011 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>

#include <boost/program_options.hpp>
#include <boost/format.hpp>
#include <boost/date_time.hpp>
#include <boost/chrono.hpp>

///////////////////////////////////////////////////////////////////////////////
int monitor(boost::uint64_t pause, boost::uint64_t values)
{
    // Resolve the GID of the performance counter using it's symbolic name.
    boost::uint32_t const prefix = hpx::applier::get_applier().get_prefix_id();        
    boost::format sine_instance("/sine(locality#%d/instance#%d)/immediate");

    using hpx::naming::id_type;
    using hpx::performance_counters::get_counter;

    id_type id1 = get_counter(boost::str(sine_instance % prefix % 0));
    id_type id2 = get_counter(boost::str(sine_instance % prefix % 1));

    using boost::chrono::duration;
    boost::int64_t start_time = 0;

    while (values-- > 0) 
    {
        // Query the performance counter.
        using hpx::performance_counters::counter_value;
        using hpx::performance_counters::status_valid_data;
        using hpx::performance_counters::stubs::performance_counter;

        counter_value value1 = performance_counter::get_value(id1); 
        counter_value value2 = performance_counter::get_value(id2); 
        if (status_valid_data == value1.status_)
        {
            if (!start_time)
                start_time = value1.time_;

            std::cout << (boost::format("%.3f: %.4f, %.4f\n") % 
                ((value1.time_ - start_time) * 10e-10) % 
                value1.get_value<double>() % 
                (value2.value_ / 100000.));
        }

        // Schedule a wakeup.
        using hpx::threads::set_thread_state;
        using hpx::threads::get_self_id;
        using hpx::threads::pending;
        using hpx::threads::wait_timeout;
        using hpx::threads::thread_priority_critical;

        set_thread_state(get_self_id(), boost::posix_time::milliseconds(pause),
            pending, wait_timeout, thread_priority_critical);
        
        // give up control to the thread manager, we will be resumed after
        // 'pause' ms as specified above
        using hpx::threads::get_self;
        using hpx::threads::suspended;

        get_self().yield(suspended);
    }
    return 0;
}

///////////////////////////////////////////////////////////////////////////////
int hpx_main(boost::program_options::variables_map& vm)
{
    // retrieve the command line arguments
    boost::uint64_t const pause = vm["pause"].as<boost::uint64_t>();
    boost::uint64_t const values = vm["values"].as<boost::uint64_t>();

    // do main work, i.e. query the performance counter
    std::cout << "starting sine monitoring..." << std::endl;
    int result = monitor(pause, values);

    // Initiate shutdown of the runtime system.
    hpx::finalize();
    return result;
}

///////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
    using boost::program_options::options_description;
    using boost::program_options::value;

    // Configure application-specific options.
    options_description desc_commandline("usage: sine_client [options]");
    desc_commandline.add_options()
            ("pause", value<boost::uint64_t>()->default_value(500),
             "milliseconds between each performance counter query")
            ("values", value<boost::uint64_t>()->default_value(100),
             "number of performance counter queries to perform")
        ;

    // Initialize and run HPX.
    return hpx::init(desc_commandline, argc, argv);
}

