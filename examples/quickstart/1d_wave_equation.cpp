////////////////////////////////////////////////////////////////////////////////
//  Copyright (c)      2012 Zach Byerly
//  Copyright (c) 2011-2012 Bryce Adelstein-Lelbach
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

//
// This is a program written to evolve in time the equation:
//
// D^2 U / Dt^2 = c^2  D^2 U / Dx^2
//
// The parameter alpha = c*dt/dx must be less than 1 to ensure the stability
//     of the algorithm.
// Discretizing the equation and solving for U(t+dt,x) yields
// alpha^2 * (U(t,x+dx)+U(t,x-dx))+2(1-alpha^2)*U(t,x) - U(t-dt,x)
//
// For the first timestep, we approximate U(t-dt,x) by u(t+dt,x)-2*dt*du/dt(t,x)
//

// Include statements.
#include <hpx/config.hpp>
#if !defined(HPX_COMPUTE_DEVICE_CODE)
#include <hpx/hpx_init.hpp>
#include <hpx/chrono.hpp>
#include <hpx/iostream.hpp>
#include <hpx/include/async.hpp>
#include <hpx/lcos/future_wait.hpp>
#include <hpx/modules/actions_base.hpp>
#include <hpx/modules/format.hpp>

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <mutex>
#include <vector>

using hpx::program_options::options_description;
using hpx::program_options::value;
using hpx::program_options::variables_map;

using hpx::naming::id_type;
using hpx::naming::invalid_id;

using hpx::async;
using hpx::lcos::future;
using hpx::lcos::wait;

using hpx::chrono::high_resolution_timer;

using hpx::find_here;

using hpx::cout;
using hpx::flush;

///////////////////////////////////////////////////////////////////////////////
// Globals.

//double const alpha_squared = 0.25;
double alpha_squared = 0;

// Initialized in hpx_main.
id_type here = invalid_id;
double pi = 0.;
double c = 0.;
double dt = 0.;
double dx = 0.;

// Command line argument.
std::uint64_t nt = 0;
std::uint64_t nx = 0;

struct data
{
    // Default constructor: data d1;
    data()
      : mtx()
      , u_value(0.0)
      , computed(false)
    {
    }

    // Copy constructor: data d1; data d2(d1);
    // We can't copy the mutex, because mutexes are noncopyable.
    data(data const& other)
      : mtx()
      , u_value(other.u_value)
      , computed(other.computed)
    {
    }

    data& operator=(data const& other)
    {
        u_value = other.u_value;
        computed = other.computed;
        return *this;
    }

    hpx::lcos::local::mutex mtx;
    double u_value;
    bool computed;
};

std::vector<std::vector<data>> u;

///////////////////////////////////////////////////////////////////////////////
// Forward declaration of the wave function.
double wave(std::uint64_t t, std::uint64_t x);

// Any global function needs to be wrapped into a plain_action if it should be
// invoked as a HPX-thread.
// This generates the required boilerplate we need for remote invocation.
HPX_PLAIN_ACTION(wave);

double calculate_u_tplus_x(
    double u_t_xplus, double u_t_x, double u_t_xminus, double u_tminus_x)
{
    double u_tplus_x = alpha_squared * (u_t_xplus + u_t_xminus) +
        2.0 * (1 - alpha_squared) * u_t_x - u_tminus_x;
    return u_tplus_x;
}

double calculate_u_tplus_x_1st(
    double u_t_xplus, double u_t_x, double u_t_xminus, double u_dot)
{
    double u_tplus_x = 0.5 * alpha_squared * (u_t_xplus + u_t_xminus) +
        (1 - alpha_squared) * u_t_x + dt * u_dot;
    return u_tplus_x;
}

double wave(std::uint64_t t, std::uint64_t x)
{
    {
        std::lock_guard<hpx::lcos::local::mutex> l(u[t][x].mtx);
        //  hpx::util::format_to(cout, "calling wave... t={1} x={2}\n", t, x);
        if (u[t][x].computed)
        {
            //cout << ("already computed!\n");
            return u[t][x].u_value;
        }
        u[t][x].computed = true;

        if (t == 0)    //first timestep are initial values
        {
            //        hpx::util::format_to(cout, "first timestep\n");
            u[t][x].u_value =
                std::sin(2. * pi * x * dx);    // initial u(x) value
            return u[t][x].u_value;
        }
    }

    // NOT using ghost zones here... just letting the stencil cross the periodic
    // boundary.
    future<double> n1;
    if (x == 0)
        n1 = async<wave_action>(here, t - 1, nx - 1);
    else
        n1 = async<wave_action>(here, t - 1, x - 1);

    future<double> n2 = async<wave_action>(here, t - 1, x);

    future<double> n3;
    if (x == (nx - 1))
        n3 = async<wave_action>(here, t - 1, 0);
    else
        n3 = async<wave_action>(here, t - 1, x + 1);

    double u_t_xminus = n1.get();    //get the futures
    double u_t_x = n2.get();
    double u_t_xplus = n3.get();

    if (t == 1)    //second time coordinate handled differently
    {
        std::lock_guard<hpx::lcos::local::mutex> l(u[t][x].mtx);
        double u_dot = 0;    // initial du/dt(x)
        u[t][x].u_value =
            calculate_u_tplus_x_1st(u_t_xplus, u_t_x, u_t_xminus, u_dot);
        return u[t][x].u_value;
    }
    else
    {
        std::lock_guard<hpx::lcos::local::mutex> l(u[t][x].mtx);
        double u_tminus_x = async<wave_action>(here, t - 2, x).get();
        u[t][x].u_value =
            calculate_u_tplus_x(u_t_xplus, u_t_x, u_t_xminus, u_tminus_x);
        return u[t][x].u_value;
    }
}

///////////////////////////////////////////////////////////////////////////////
int hpx_main(variables_map& vm)
{
    here = find_here();
    pi = 3.141592653589793238462643383279;

    //    dt = vm["dt-value"].as<double>();
    //    dx = vm["dx-value"].as<double>();
    //    c = vm["c-value"].as<double>();
    nx = vm["nx-value"].as<std::uint64_t>();
    nt = vm["nt-value"].as<std::uint64_t>();

    c = 1.0;

    dt = 1.0 / static_cast<double>(nt - 1);
    dx = 1.0 / static_cast<double>(nx - 1);
    alpha_squared = (c * dt / dx) * (c * dt / dx);

    // check that alpha_squared satisfies the stability condition
    if (0.25 < alpha_squared)
    {
        cout << (("alpha^2 = (c*dt/dx)^2 should be less than 0.25 for "
                  "stability!\n"))
             << flush;
    }

    u = std::vector<std::vector<data>>(nt, std::vector<data>(nx));

    hpx::util::format_to(cout, "dt = {1}\n", dt) << flush;
    hpx::util::format_to(cout, "dx = {1}\n", dx) << flush;
    hpx::util::format_to(cout, "alpha^2 = {1}\n", alpha_squared) << flush;

    {
        // Keep track of the time required to execute.
        high_resolution_timer t;

        std::vector<future<double>> futures;
        for (std::uint64_t i = 0; i < nx; i++)
            futures.push_back(async<wave_action>(here, nt - 1, i));

        // open file for output
        std::ofstream outfile;
        outfile.open("output.dat");

        auto f = [&](std::size_t i, double n) {
            double x_here = i * dx;
            hpx::util::format_to(outfile, "{1} {2}\n", x_here, n) << flush;
        };
        wait(futures, f);

        outfile.close();

        char const* fmt = "elapsed time: {1} [s]\n";
        hpx::util::format_to(std::cout, fmt, t.elapsed());
    }

    hpx::finalize();
    return 0;
}

///////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
    // Configure application-specific options.
    options_description desc_commandline(
        "Usage: " HPX_APPLICATION_STRING " [options]");

    // clang-format off
    desc_commandline.add_options()
        ("dt-value",
           value<double>()->default_value(0.05),
           "dt parameter of the wave equation")

        ("dx-value",
           value<double>()->default_value(0.1),
           "dx parameter of the wave equation")

        ("c-value",
           value<double>()->default_value(1.0),
           "c parameter of the wave equation")

        ("nx-value",
           value<std::uint64_t>()->default_value(100),
           "nx parameter of the wave equation")

        ("nt-value",
           value<std::uint64_t>()->default_value(100),
           "nt parameter of the wave equation")
    ;
    // clang-format on

    // Initialize and run HPX.
    hpx::init_params init_args;
    init_args.desc_cmdline = desc_commandline;

    return hpx::init(argc, argv, init_args);
}
#endif
