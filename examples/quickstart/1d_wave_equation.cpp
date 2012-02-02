////////////////////////////////////////////////////////////////////////////////
//  Copyright (c)      2012 Zach Byerly
//  Copyright (c) 2011-2012 Bryce Adelstein-Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

// Include statements.
#include <hpx/hpx_init.hpp>
#include <hpx/runtime/actions/plain_action.hpp>
#include <hpx/runtime/components/plain_component_factory.hpp>
#include <hpx/lcos/async.hpp>
#include <hpx/lcos/async.hpp>
#include <hpx/lcos/async_future_wait.hpp>
#include <hpx/include/iostreams.hpp>

#include <boost/format.hpp>
#include <boost/math/constants/constants.hpp>

using boost::program_options::variables_map;
using boost::program_options::options_description;
using boost::program_options::value;

using hpx::naming::id_type;
using hpx::naming::invalid_id;

using hpx::actions::plain_result_action2;

using hpx::lcos::promise;
using hpx::lcos::async;
using hpx::lcos::wait;
using hpx::lcos::eager_future;

using hpx::init;
using hpx::finalize;
using hpx::find_here;

using hpx::cout;
using hpx::flush;

///////////////////////////////////////////////////////////////////////////////
// Globals.

double const alpha_squared = 0.25;

// Initialized in hpx_main.
id_type here = invalid_id; 
double pi = 0.; 

// Command line argument.
double dt = 0.; 
double dx = 0.;
double c = 0.;
boost::uint64_t nx = 0;
boost::uint64_t nt = 0;
 
///////////////////////////////////////////////////////////////////////////////
// Forward declaration of the wave function.
double wave(boost::uint64_t t, boost::uint64_t x);

// Any global function needs to be wrapped into a plain_action if it should be
// invoked as a HPX-thread.
typedef plain_result_action2<
    // result type
    double,             
    
    // arguments
    boost::uint64_t,    
    boost::uint64_t,

    // function
    wave
> wave_action;

// This generates the required boilerplate we need for remote invocation.
HPX_REGISTER_PLAIN_ACTION(wave_action);

///////////////////////////////////////////////////////////////////////////////
// An eager_future is a HPX construct exposing the semantics of a Future
// object. It starts executing the bound action immediately (eagerly).
typedef eager_future<wave_action> wave_future;

double calculate_u_tplus_x(double u_t_xplus, double u_t_x, double u_t_xminus,
    double u_tminus_x)
{
    double u_tplus_x = alpha_squared*(u_t_xplus + u_t_xminus)
              + 2.0*(1-alpha_squared)*u_t_x - u_tminus_x;
    return u_tplus_x;
}

double calculate_u_tplus_x_1st(double u_t_xplus, double u_t_x,
    double u_t_xminus, double u_dot)
{
    double u_tplus_x = alpha_squared*(u_t_xplus + u_t_xminus)
              + 2.0*(1-alpha_squared)*u_t_x + dt*u_dot;
    return u_tplus_x;
}

double wave(boost::uint64_t t, boost::uint64_t x)
{
    if (t == 0) //first timestep are initial values
        return std::sin(2*pi*x); // initial u(x) value

    promise<double> n1;

    // NOT using ghost zones here... just letting the stencil cross the periodic
    // boundary.
    if (x == 0)
        n1 = async<wave_action>(here,t-1,nx-1);
    else
        n1 = async<wave_action>(here,t-1,x-1);

    wave_future n2(here,t-1,x);

    promise<double> n3;

    if (x == (nx-1))
        n3 = async<wave_action>(here,t-1,0);
    else
        n3 = async<wave_action>(here,t-1,x+1);

    double u_t_xminus = n1.get(); //get the futures
    double u_t_x = n2.get();
    double u_t_xplus = n3.get();

    if (t == 1) //second time coordinate handled differently
    {
        double u_dot = 0;// initial du/dt(x)
        return calculate_u_tplus_x_1st(u_t_xplus,u_t_x,u_t_xminus,u_dot);
    } else {
        wave_future n4(here,t-2,x);
        double u_tminus_x = n4.get();
        return calculate_u_tplus_x(u_t_xplus,u_t_x,u_t_xminus,u_tminus_x);
    }
}

///////////////////////////////////////////////////////////////////////////////
int hpx_main(variables_map& vm)
{
    here = find_here();
    pi = boost::math::constants::pi<double>();

    dt = vm["dt-value"].as<double>();
    dx = vm["dx-value"].as<double>();
    c = vm["c-value"].as<double>();
    nx = vm["nx-value"].as<boost::uint64_t>();
    nt = vm["nt-value"].as<boost::uint64_t>();

    {
        std::vector<promise<double> > futures;

        for (boost::uint64_t i=0;i<nx;i++) 
            futures.push_back(async<wave_action>(here,i,nt));

        wait(futures, [&](std::size_t i, double n)
            { cout << (boost::format("u[%1%] = %2%") % i % n) << flush; });
    }

    finalize();
    return 0;
}

///////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
    // Configure application-specific options.
    options_description
       desc_commandline("Usage: " HPX_APPLICATION_STRING " [options]");

    desc_commandline.add_options()
        ( "dt-value"
        , value<double>()->default_value(0.1)
        , "dt parameter of the wave equation")

        ( "dx-value"
        , value<double>()->default_value(0.1)
        , "dx parameter of the wave equation")

        ( "c-value"
        , value<double>()->default_value(0.5)
        , "c parameter of the wave equation")

        ( "nx-value"
        , value<boost::uint64_t>()->default_value(10)
        , "nx parameter of the wave equation")

        ( "nt-value"
        , value<boost::uint64_t>()->default_value(10)
        , "nt parameter of the wave equation")
        ;

    // Initialize and run HPX.
    return init(desc_commandline, argc, argv);
}

