//  Copyright (c) 2007-2011 Hartmut Kaiser
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
//     std::cout << "elapsed: " << elapsed << ", result: " << result << std::endl;
// }

///////////////////////////////////////////////////////////////////////////////
int fib(naming::id_type prefix, int n, int delay_coeff);

typedef 
    actions::plain_result_action3<int, naming::id_type, int, int, fib> 
fibonacci1_action;

HPX_REGISTER_PLAIN_ACTION(fibonacci1_action);

///////////////////////////////////////////////////////////////////////////////
int count_invocations = 0;    // global invocation counter

int fib (naming::id_type that_prefix, int n, int delay_coeff)
{
    // count number of invocations
    ++count_invocations;

    // do some busy waiting, if requested
    if (delay_coeff) {
        util::high_resolution_timer t;
        double start_time = t.elapsed();
        double current = 0;
        do {
            current = t.elapsed();
        } while (current - start_time < delay_coeff * 1e-6);
    }

    // here is the actual fibonacci calculation
    if (n < 2) 
        return n;

    typedef lcos::eager_future<fibonacci1_action> fibonacci1_future;

    // execute the first fib() at the other locality, returning here afterwards
    // execute the second fib() here, forwarding the correct prefix
    naming::id_type this_prefix = applier::get_applier().get_runtime_support_gid();
    fibonacci1_future n1(that_prefix, this_prefix, n - 1, delay_coeff);
    fibonacci1_future n2(this_prefix, that_prefix, n - 2, delay_coeff);

//     std::cout << "*";

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

    // try to get arguments from application configuration
    runtime& rt = get_runtime();
    argument = boost::lexical_cast<int>(
        rt.get_config().get_entry("application.fibonacci1.argument", argument));

    // get list of all known localities
    std::vector<naming::id_type> prefixes;
    applier::applier& appl = applier::get_applier();

    naming::id_type this_prefix = appl.get_runtime_support_gid();
    naming::id_type that_prefix;
    components::component_type type = 
        components::get_component_type<components::server::plain_function<fibonacci1_action> >();

    if (appl.get_remote_prefixes(prefixes, type)) {
        // execute the fib() function on any of the remote localities
        that_prefix = prefixes[0];
    }
    else {
        // execute the fib() function locally
        that_prefix = this_prefix;
    }

    {
        util::high_resolution_timer t;
        lcos::eager_future<fibonacci1_action> n(
            that_prefix, this_prefix, argument, delay_coeff);
        result = n.get();
        elapsed = t.elapsed();
    }

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

///////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
    // Configure application-specific options
    po::options_description 
        desc_commandline ("Usage: fibonacci1 [options]");
    desc_commandline.add_options()
        ("value,v", po::value<int>(), 
         "the number to be used as the argument to fib (default is 10)")
        ("csv,s", "generate statistics of the run in comma separated format")
        ("busywait,b", po::value<int>(),
         "add this amount of busy wait workload to each of the iterations"
         " [in microseconds], i.e. -b1000 == 1 millisecond")
        ;

    // Initialize and run HPX
    int retcode = hpx::init(desc_commandline, argc, argv);
    return retcode;
}
