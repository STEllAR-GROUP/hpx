//  Copyright (c) 2010-2011 Phillip LeBlanc, Hartmut Kaiser
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

int fib(int n, int delay_coeff);
int fib_rhs(int n, int delay_coeff);

typedef 
    actions::plain_result_action2<int, int, int, fib> 
fibonacci4_action;

typedef 
    actions::plain_result_action2<int, int, int, fib_rhs> 
fibonacci4_rhs_action;


typedef lcos::eager_future<fibonacci4_action> fibonacci_future;
typedef lcos::eager_future<fibonacci4_rhs_action> fibonacci_rhs_future;

HPX_REGISTER_PLAIN_ACTION(fibonacci4_action);
HPX_REGISTER_PLAIN_ACTION(fibonacci4_rhs_action);

///////////////////////////////////////////////////////////////////////////////
int count_invocations = 0;    // global invocation counter
inline void count_invocation(void) { ++count_invocations; }

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


int fib (int n, int delay_coeff)
{
    // count number of invocations
    count_invocation();

    // do some busy waiting, if requested
    do_busy_work(delay_coeff);

    // here is the actual fibonacci calculation
    if (n < 2) 
        return n;

    id_type here = find_here();
    id_type next = get_runtime().get_process().next();

    fibonacci_future n1(next, n - 1, delay_coeff);
    fibonacci_rhs_future n2(here,  n - 2, delay_coeff);
    return n1.get() + n2.get();

}

int fib_rhs (int n, int delay_coeff)
{
    // count number of invocations
    count_invocation();

    // do some busy waiting, if requested
    do_busy_work(delay_coeff);

    // here is the actual fibonacci calculation
    if (n < 2) 
        return n;

    id_type here = find_here();

    fibonacci_rhs_future n1(here,  n - 1, delay_coeff);
    fibonacci_rhs_future n2(here,  n - 2, delay_coeff);

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
    get_option(vm, "value", argument, "application.fibonacci2.argument");
    get_option(vm, "busywait", delay_coeff);

    gid_type here = find_here();

    {
        util::high_resolution_timer t;
        fibonacci_future n(here, argument, delay_coeff);
        result = n.get();
        elapsed = t.elapsed();
    }

    // Write results
    std::cout << "elapsed: " << elapsed << ", result: " << result << std::endl;
    std::cout << "Number of invocations of fib(): " << count_invocations 
                                                       << std::endl;

    // initiate shutdown of the runtime systems on all localities
    components::stubs::runtime_support::shutdown_all();

    return 0;
}


int main(int argc, char* argv[])
{
  // Configure application-specific options
  po::options_description 
      desc_commandline("Usage: fibonacci4 [hpx_options] [options]");
  desc_commandline.add_options()
      ("value,v", po::value<int>(), 
       "the number to be used as the argument to fib (default is 10)")
      ("busywait,b", po::value<int>(),
       "add this amount of busy wait workload to each of the iterations"
       " [in steps of 1µs], i.e. -b1000 == 1ms")
      ;

  // Initialize and run HPX
  int retcode = hpx_init(desc_commandline, argc, argv); 
  return retcode;
}
