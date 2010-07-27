//  Copyright (c) 2007-2010 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <iostream>

#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>

#include <hpx/runtime/actions/plain_action.hpp>
#include <hpx/runtime/components/plain_component_factory.hpp>

#include <boost/program_options.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/export.hpp>

using namespace hpx;
namespace po = boost::program_options;

///////////////////////////////////////////////////////////////////////////////
// Helpers
typedef hpx::naming::gid_type gid_type;

///////////////////////////////////////////////////////////////////////////////
// int fib(int n)
// {
//     if (n < 2) 
//         return n;
// 
//     int n1 = fib(n - 1);
//     int n2 = fib(n - 2);
//     return n1 + n2;
// }
// 
// int main()
// {
//     util::high_resolution_timer t;
//     int result = fib(41);
//     double elapsed = t.elapsed();
// 
//     std::cout << "elapsed: " << elapsed 
//               << ", result: " << result << std::endl;
// }

///////////////////////////////////////////////////////////////////////////////
int fib(gid_type prefix, int n, int delay_coeff);

typedef 
    actions::plain_result_action3<int, gid_type, int, int, fib> 
fibonacci2_action;

HPX_REGISTER_PLAIN_ACTION(fibonacci2_action);

///////////////////////////////////////////////////////////////////////////////
typedef lcos::eager_future<fibonacci2_action> fibonacci_future;

///////////////////////////////////////////////////////////////////////////////
inline void do_busy_work(double delay_coeff)
{
    if (delay_coeff) {
        util::high_resolution_timer t;
        double start_time = t.elapsed();
        double current = 0;
        do {
            current = t.elapsed();
        } while (current - start_time < delay_coeff * 1e-6);
    }
}

int fib (gid_type there, int n, int delay_coeff)
{
    // do some busy waiting, if requested
    do_busy_work(delay_coeff);

    // here is the actual fibonacci calculation
    if (n < 2) 
        return n;

    // execute the first fib() at the other locality, returning here afterwards
    // execute the second fib() here, forwarding the correct prefix
    gid_type here = get_runtime().here();
    fibonacci_future n1(there, here, n - 1, delay_coeff);
    fibonacci_future n2(here, there, n - 2, delay_coeff);

    return n1.get() + n2.get();
}

///////////////////////////////////////////////////////////////////////////////
int hpx_main(po::variables_map &vm)
{
    int argument = 10;
    int delay_coeff = 0;
    int result = 0;
    double elapsed = 0.0;

    // Process application-specific command-line options
    if (vm.count("value"))
        argument = vm["value"].as<int>();
    if (vm.count("busywait"))
        delay_coeff = vm["busywait"].as<int>();

    // Try to get arguments from application configuration
    runtime& rt = get_runtime();
    argument = boost::lexical_cast<int>(
        rt.get_config().get_entry("application.fibonacci2.argument", argument));

    gid_type here = get_runtime().here();
    gid_type there = get_runtime().next();

    {
        util::high_resolution_timer t;
        fibonacci_future n(there, here, argument, delay_coeff);
        result = n.get();
        elapsed = t.elapsed();
    }

    LAPP_(info) << "Elapsed time: " << elapsed;
    LAPP_(info) << "Result: " << result;

    if (vm.count("csv"))
    {
      // write results as csv
      std::cout << argument << "," 
        << elapsed << "," << result << "," << std::endl;
    }
    else {
      // write results the old fashioned way
      std::cout << "elapsed: " << elapsed << ", result: " << result << std::endl;
    }

    // initiate shutdown of the runtime systems on all localities
    components::stubs::runtime_support::shutdown_all();

    return 0;
}

int main(int argc, char* argv[])
{
  // Configure application-specific options
  po::options_description 
      desc_commandline("Usage: fibonacci2 [hpx_options] [options]");
  desc_commandline.add_options()
      ("value,v", po::value<int>(), 
       "the number to be used as the argument to fib (default is 10)")
      ("csv,s", "generate statistics of the run in comma separated format")
      ("busywait,b", po::value<int>(),
       "add this amount of busy wait workload to each of the iterations"
       " [in steps of 1µs], i.e. -b1000 == 1ms")
      ;

  // Initialize and run HPX
  int retcode = hpx_init(desc_commandline, argc, argv); 
  return retcode;
}

