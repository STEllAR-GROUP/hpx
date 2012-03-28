/////////////////////////// Intrest Calculor //////////////////////////////////
//  Copyright (c) 2012 Adrian Serio
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//
// Calculate interest while using dataflow objects
///////////////////////////////////////////////////////////////////////////////

// When using the dataflow component we have to define the following constants
#define HPX_ACTION_ARGUMENT_LIMIT 6             // 4 more than max number of arguments
#define HPX_COMPONENT_CREATE_ARGUMENT_LIMIT 6   // 4 more than max number of arguments
#define HPX_FUNCTION_LIMIT 9                    // 7 more than max number of arguments

#include <iostream>

#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/runtime/actions/plain_action.hpp>
#include <hpx/components/dataflow/dataflow.hpp>

#include <hpx/runtime/components/plain_component_factory.hpp>

using hpx::actions::plain_result_action1;
using hpx::actions::plain_result_action2;
using boost::program_options::variables_map;
using boost::program_options::options_description;
using boost::program_options::value;

///////////////////////////////////////////////////////////////////////////////
// Calculate interest for one period
double calc(double principal, double rate)
{
    return principal * rate;
}

///////////////////////////////////////////////////////////////////////////////
// Add the amount made to the principal
double add(double principal, double interest)
{
    return principal + interest;
}

///////////////////////////////////////////////////////////////////////////////
// Action Declarations
typedef plain_result_action2<
    double,    // return type
    double,    // arguments
    double,
    &calc      // wrapped function
> calc_action;

HPX_REGISTER_PLAIN_ACTION(calc_action);

typedef plain_result_action2<
    double,    // return type
    double,    // arguments
    double,
    &add       // wrapped function
> add_action;

HPX_REGISTER_PLAIN_ACTION(add_action);

///////////////////////////////////////////////////////////////////////////////
// This is a helper function allowing to encapsulate the initial values into a
// dataflow object
double identity(double initial_value)
{
    return initial_value;
}

typedef plain_result_action1<
    double,
    double,
    &identity
> identity_action;

HPX_REGISTER_PLAIN_ACTION(identity_action);

///////////////////////////////////////////////////////////////////////////////
int hpx_main(variables_map & vm)
{
    using hpx::lcos::dataflow;
    using hpx::lcos::dataflow_base;
    hpx::naming::id_type here = hpx::find_here;

    // In non-dataflow terms the implemented algorithm would look like:
    //
    // int t = 5;    // number of time periods to use
    // double principal = 1e6;
    // double rate = 0.20;
    //
    // for (int i = 0; i < t; ++i)
    // {
    //     double interest = calc(principal, rate);
    //     principal = add(principal, interest);
    // }
    //
    // Please note the similarity with the code below!

    int t = 5;    // number of time periods to use
    dataflow_base<double> principal = dataflow<identity_action>(here, 1e6);
    dataflow_base<double> rate = dataflow<identity_action>(here, 0.20);

    for (int i = 0; i < t; ++i)
    {
        dataflow_base<double> interest = dataflow<calc_action>(here, principal, rate);
        principal = dataflow<add_action>(here, principal, interest);
    }

    // wait for the dataflow execution graph to be finished calculating our
    // overall interest
    double result = principal.get_future().get();


    std::cout << "Final amount: " << result << std::endl;
    std::cout << "Amount made: " << result-1e6 << std::endl;
    return hpx::finalize();
}

///////////////////////////////////////////////////////////////////////////////
int main(int argc, char ** argv)
{
    options_description cmdline("Usage: " HPX_APPLICATION_STRING " [options]");

    cmdline.add_options()
        ("principal", value<double>(), "The principal [$]")
        ("rate", value<double>(), "The intrest rate [%]")
        ("cp", value<int>(), "The compound period [months]")
        ("time", value<int>(), "The time money is invested [months]")
    ;

    return hpx::init(cmdline, argc, argv);
}

