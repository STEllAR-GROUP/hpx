//  Copyright (c) 2014 Hartmut Kaiser
//  Copyright (c) 2014 Bryce Adelstein-Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// This is the second in a series of examples demonstrating the development of
// a fully distributed solver for a simple 1D heat distribution problem.
//
// This example shows how futurization can be applied to the code from example
// one. While this nicely parallelizes the code (note: without changing the
// overall structure of the algorithm), the achieved performance is bad (a lot
// slower than example one). This is caused by the large amount of overheads
// introduced by wrapping each and every grid point into its own future object.
// The amount of work performed by each of the created HPX threads (one thread
// for every grid point and time step) is too small compared to the imposed
// overheads.

#include <hpx/hpx_init.hpp>
#include <hpx/hpx.hpp>

///////////////////////////////////////////////////////////////////////////////
double k = 0.5;     // heat transfer coefficient
double dt = 1.;     // time step
double dx = 1.;     // grid spacing

///////////////////////////////////////////////////////////////////////////////
typedef hpx::shared_future<double> partition;
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
    using hpx::lcos::local::dataflow;
    using hpx::util::unwrapped;

    boost::uint64_t nt = vm["nt"].as<boost::uint64_t>();   // Number of steps.
    boost::uint64_t nx = vm["nx"].as<boost::uint64_t>();   // Number of grid points.

    // U[t][i] is the state of position i at time t.
    std::vector<space> U(2);
    for (space& s : U)
        s.resize(nx);

    // Initial conditions:
    //   f(0, i) = i
    for (std::size_t i = 0; i != nx; ++i)
        U[0][i] = hpx::make_ready_future(double(i));

    // Our operator:
    //   f(t+1, i) = (f(t, i-1) + f(t, i) + f(t, i+1)) / 3
    auto Op = unwrapped(heat);

    for (std::size_t t = 0; t != nt; ++t)
    {
        space& current = U[t % 2];
        space& next = U[(t + 1) % 2];

        // WHEN U[t][i-1], U[t][i], and U[t][i+1] have been computed, THEN we
        // can compute U[t+1][i]
        for (std::size_t i = 0; i != nx; ++i)
            next[i] = dataflow(Op, current[idx(i-1, nx)], current[i], current[idx(i+1, nx)]);
    }

    // Now the asynchronous computation is running; the above for-loop does not
    // wait on anything. There is no implicit waiting at the end of each timestep;
    // the computation of each U[t][i] will begin when as soon as its dependencies
    // are ready and hardware is available.

    // Wait for the solution at time-step 'nt'.
    space solution = hpx::when_all(U[nt % 2]).get();

    // Print the solution at time-step 'nt'.
    for (std::size_t i = 0; i != nx; ++i)
        std::cout << "U[" << i << "] = " << solution[i].get() << std::endl;

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

