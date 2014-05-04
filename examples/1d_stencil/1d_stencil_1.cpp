//  Copyright (c) 2014 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// This is the first in a series of examples demonstrating the development of a
// fully distributed solver for a simple 1D heat distribution problem.
//
// This example provides a serial base line implementation. No parallelization
// is performed.

#include <hpx/hpx_init.hpp>
#include <hpx/hpx.hpp>

///////////////////////////////////////////////////////////////////////////////
double k = 0.5;     // heat transfer coefficient
double dt = 1.;     // time step
double dx = 1.;     // grid spacing

///////////////////////////////////////////////////////////////////////////////
typedef double partition;
typedef std::vector<partition> space;

///////////////////////////////////////////////////////////////////////////////
inline std::size_t idx(std::size_t i, std::size_t size)
{
    return (boost::int64_t(i) < 0) ? (i + size) % size : i % size;
}

///////////////////////////////////////////////////////////////////////////////
// Our operator:
inline double heat(double left, double middle, double right)
{
    return middle + (k*dt/dx*dx) * (left - 2*middle + right);
}

///////////////////////////////////////////////////////////////////////////////
int hpx_main(boost::program_options::variables_map& vm)
{
    boost::uint64_t nt = vm["nt"].as<boost::uint64_t>();   // Number of steps.
    boost::uint64_t nx = vm["nx"].as<boost::uint64_t>();   // Number of grid points.

    // U[t][i] is the state of position i at time t.
    std::vector<space> U(2);
    for (space& s : U)
        s.resize(nx);

    // Initial conditions:
    //   f(0, i) = i
    for (std::size_t i = 0; i != nx; ++i)
        U[0][i] = double(i);

    for (std::size_t t = 0; t != nt; ++t)
    {
        space const& current = U[t % 2];
        space& next = U[(t + 1) % 2];

        for (std::size_t i = 0; i != nx; ++i)
            next[i] = heat(current[idx(i-1, nx)], current[i], current[idx(i+1, nx)]);
    }

    // Print the solution at time-step 'nt'.
    space const& solution = U[nt % 2];

    for (std::size_t i = 0; i != nx; ++i)
        std::cout << "U[" << i << "] = " << solution[i] << std::endl;

    return hpx::finalize();
}

int main(int argc, char* argv[])
{
    using namespace boost::program_options;

    options_description desc_commandline;
    desc_commandline.add_options()
        ("nx", value<boost::uint64_t>()->default_value(100),
         "Local x dimension")
        ("nt", value<boost::uint64_t>()->default_value(45),
         "Number of time steps")
        ("k", value<double>(&k)->default_value(0.5),
         "Heat transfer coefficient (default: 0.5)")
        ("dt", value<double>(&dt)->default_value(1.0),
         "Timestep unit (default: 1.0[s])")
        ("dx", value<double>(&dx)->default_value(1.0),
         "Local x dimension")
    ;

    // Initialize and run HPX
    return hpx::init(desc_commandline, argc, argv);
}
