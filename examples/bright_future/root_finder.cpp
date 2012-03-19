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
#include <hpx/lcos/async.hpp>

using boost::program_options::variables_map;
using boost::program_options::options_description;
using boost::program_options::value;

using hpx::naming::id_type;
using hpx::actions::plain_result_action2;
using hpx::lcos::async;
using hpx::util::high_resolution_timer;
using hpx::init;
using hpx::finalize;
using hpx::find_here;

double update(double x, double x_n);

typedef
    plain_result_action2<
        double // result type
      , double // arg1
      , double // arg2
      , update // function
    >
    update_action;

HPX_REGISTER_PLAIN_ACTION(update_action);

double update(double x, double x_n)
{
    return (x_n * x_n + x)/(2.0 * x_n);
}

int hpx_main(variables_map & vm)
{

    double x = vm["x"].as<double>();
    unsigned max_iterations = vm["max_iterations"].as<unsigned>();

    std::cout << "Single core version without futures:\n";

    high_resolution_timer t;
    double x_n = x;

    for(unsigned i = 0; i < max_iterations; ++i)
    {
        double x_new = update(x, x_n);
        if(std::abs(x_n - x_new) < 1e-10) break;
        x_n = x_new;
    }

    double time_elapsed = t.elapsed();

    std::cout
        << "sqrt(" << x << ") = " << x_n
        << " (calculated in " << time_elapsed << " seconds)\n"
        ;

    std::cout << "Naive future version:\n";

    t.restart();
    x_n = x;

    for(unsigned i = 0; i < max_iterations; ++i)
    {
        double x_new = async<update_action>(find_here(), x, x_n).get();
        if(std::abs(x_n - x_new) < 1e-10) break;
        x_n = x_new;
    }

    time_elapsed = t.elapsed();

    std::cout
        << "sqrt(" << x << ") = " << x_n
        << " (calculated in " << time_elapsed << " seconds)\n"
        ;

    finalize();
    return 0;
}

int main(int argc, char **argv)
{
    options_description
        desc_commandline("usage: " HPX_APPLICATION_STRING " [options]");

    desc_commandline.add_options()
        ( "x", value<double>()->default_value(2.0), "Find the square root of x")
        ( "max_iterations", value<unsigned>()->default_value(1000), "Maximum number of iterations")
        ;

    return init(desc_commandline, argc, argv);
}
