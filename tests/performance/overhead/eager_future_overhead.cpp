//  Copyright (c) 2011 Bryce Adelstein-Lelbach 
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <stdexcept>

#include <boost/format.hpp>
#include <boost/bind.hpp>
#include <boost/cstdint.hpp>
#include <boost/asio/deadline_timer.hpp>

#include <hpx/runtime.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/runtime/actions/plain_action.hpp>
#include <hpx/runtime/components/plain_component_factory.hpp>
#include <hpx/util/high_resolution_timer.hpp>
#include <hpx/lcos/eager_future.hpp>

using boost::posix_time::seconds;

using boost::asio::deadline_timer;

using boost::program_options::variables_map;
using boost::program_options::options_description;
using boost::program_options::value;

using hpx::init;
using hpx::finalize;

using hpx::find_here;
using hpx::get_runtime;

using hpx::naming::id_type;

using hpx::actions::plain_result_action0;

using hpx::lcos::eager_future;

using hpx::util::high_resolution_timer;

///////////////////////////////////////////////////////////////////////////////
boost::uint64_t null_function()
{
    return 1;
}

typedef plain_result_action0<
    // result type
    boost::uint64_t
    // function
  , null_function
> null_action;

HPX_REGISTER_PLAIN_ACTION(null_action);

typedef eager_future<null_action> null_future;

///////////////////////////////////////////////////////////////////////////////
void timeout_handler(
    bool& flag
  , boost::system::error_code const&
    )
{
    flag = true;     
}
    
///////////////////////////////////////////////////////////////////////////////
int hpx_main(variables_map& vm)
{
    {
        const boost::uint64_t duration = vm["duration"].as<boost::uint64_t>();

        if (duration == 0)
            throw std::logic_error("error: duration of 0 seconds specified\n");

        const id_type here = find_here();

        bool flag = false;
        boost::uint64_t futures = 0;

        deadline_timer t( get_runtime().get_io_pool().get_io_service()
                        , seconds(duration));

        t.async_wait(boost::bind( &timeout_handler
                                , boost::ref(flag)
                                , boost::asio::placeholders::error));

        high_resolution_timer real_clock;

        while (HPX_LIKELY(!flag))
        {
            null_future nf(here);
            futures += nf.get(); 
        }

        double actual_seconds = real_clock.elapsed();

        if (vm.count("csv"))
            std::cout << ( boost::format("%1%,%2%\n")
                         % futures
                         % actual_seconds);
        else
            std::cout << ( boost::format("invoked %1% futures in %2% seconds\n")
                         % futures
                         % actual_seconds);
    }

    finalize();
    return 0;
}

///////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
    // Configure application-specific options.
    options_description cmdline("usage: " HPX_APPLICATION_STRING " [options]");

    cmdline.add_options()
        ( "duration"
        , value<boost::uint64_t>()->default_value(5) 
        , "duration of the test period in seconds")

        ( "csv"
        , "output results in csv format (number of futures invoked, test "
          "duration in seconds)")
        ;

    // Initialize and run HPX.
    return init(cmdline, argc, argv);
}

