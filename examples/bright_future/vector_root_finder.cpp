//  Copyright (c) 2011 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)


#include <hpx/hpx.hpp>

#include <iostream>
#include <hpx/hpx_init.hpp>
#include <hpx/runtime/actions/plain_action.hpp>
#include <hpx/runtime/components/plain_component_factory.hpp>
#include <hpx/util/high_resolution_timer.hpp>
#include <hpx/lcos/eager_future.hpp>
#include <hpx/include/iostreams.hpp>

using boost::program_options::variables_map;
using boost::program_options::options_description;
using boost::program_options::value;

using hpx::naming::id_type;
using hpx::actions::plain_result_action2;
using hpx::actions::plain_result_action3;
using hpx::lcos::eager_future;
using hpx::lcos::promise;
using hpx::util::high_resolution_timer;
using hpx::init;
using hpx::finalize;
using hpx::find_here;
using hpx::cout;
using hpx::flush;
using hpx::get_os_thread_count;

double update(double x, double x_n)
{
    return (x_n * x_n + x)/(2.0 * x_n);
}

double start_iteration(double x, double x_n, unsigned max_iterations)
{
    for(unsigned i = 0; i < max_iterations; ++i)
    {
        double x_new = update(x, x_n);
        //if(std::abs(x_n - x_new) < 1e-10) break;
        x_n = x_new;
    }
    return x_n;
}

typedef
    plain_result_action3<
        double // result type
      , double // arg1
      , double // arg2
      , unsigned // arg3
      , start_iteration // function
    >
    start_iteration_action;

HPX_REGISTER_PLAIN_ACTION(start_iteration_action);

typedef eager_future<start_iteration_action> start_iteration_future;

int hpx_main(variables_map & vm)
{
    unsigned vector_size = vm["vector_size"].as<unsigned>();
    unsigned max_iterations = vm["max_iterations"].as<unsigned>();

    std::vector<double> xs(vector_size);

    //std::generate(xs.begin(), xs.end(), std::rand);

    //std::cout << "Single core version without futures:\n";

    high_resolution_timer t;
    /*
    for(std::vector<double>::iterator it = xs.begin(); it != xs.end(); ++it)
    {
        double x = *it;
        double x_n = *it;

        for(unsigned i = 0; i < max_iterations; ++i)
        {
            double x_new = update(x, x_n);
            if(std::abs(x_n - x_new) < 1e-10) break;
            x_n = x_new;
        }
        *it = x_n;
    }

    double time_elapsed = t.elapsed();

    std::cout
        //<< "sqrt(" << x << ") = " << x_n
        << " (calculated in " << time_elapsed << " seconds)\n"
        ;
    */

    std::generate(xs.begin(), xs.end(), std::rand);
    //std::cout << "Naive future version:\n";

    t.restart();

    std::vector<promise<double> > promises;
    promises.reserve(xs.size());
    for(std::vector<double>::iterator it = xs.begin(); it != xs.end(); ++it)
    {
        promises.push_back(start_iteration_future(find_here(), *it, *it, max_iterations));
    }
    for(unsigned i = 0; i < vector_size; ++i)
    {
        xs[i] = promises[i].get();
    }

    double time_elapsed = t.elapsed();

    cout
        //<< "sqrt(" << x << ") = " << x_n
        //<< " (calculated in " << time_elapsed << " seconds)\n"
        << get_os_thread_count() << " " << time_elapsed << "\n" << flush
        ;

    finalize();
    return 0;
}

int main(int argc, char **argv)
{
    options_description
        desc_commandline("usage: " HPX_APPLICATION_STRING " [options]");

    desc_commandline.add_options()
        ( "vector_size", value<unsigned>()->default_value(10), "Find the square root of x")
        ( "max_iterations", value<unsigned>()->default_value(1000), "Maximum number of iterations")
        ;

    return init(desc_commandline, argc, argv);
}
