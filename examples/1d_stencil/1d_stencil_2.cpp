//  Copyright (c) 2014 Hartmut Kaiser
//  Copyright (c) 2014 Bryce Adelstein-Lelbach
//  Copyright (c) 2014 Patricia Grubel
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

#include "print_time_results.hpp"

///////////////////////////////////////////////////////////////////////////////
// Command-line variables
bool header = true; // print csv heading
double k = 0.5;     // heat transfer coefficient
double dt = 1.;     // time step
double dx = 1.;     // grid spacing

inline std::size_t idx(std::size_t i, int dir, std::size_t size)
{
    if(i == 0 && dir == -1)
        return size-1;
    if(i == size-1 && dir == +1)
        return 0;

    HPX_ASSERT((i + dir) < size);

    return i + dir;
}

///////////////////////////////////////////////////////////////////////////////
//[stepper_2
struct stepper
{
    // Our partition type
    typedef hpx::shared_future<double> partition;

    // Our data for one time step
    typedef std::vector<partition> space;

    // Our operator
    static double heat(double left, double middle, double right)
    {
        return middle + (k*dt/(dx*dx)) * (left - 2*middle + right);
    }

    // do all the work on 'nx' data points for 'nt' time steps
    hpx::future<space> do_work(std::size_t nx, std::size_t nt)
    {
        using hpx::dataflow;
        using hpx::util::unwrapped;

        // U[t][i] is the state of position i at time t.
        std::vector<space> U(2);
        for (space& s : U)
            s.resize(nx);

        // Initial conditions: f(0, i) = i
        for (std::size_t i = 0; i != nx; ++i)
            U[0][i] = hpx::make_ready_future(double(i));

        auto Op = unwrapped(&stepper::heat);

        // Actual time step loop
        for (std::size_t t = 0; t != nt; ++t)
        {
            space const& current = U[t % 2];
            space& next = U[(t + 1) % 2];

            // WHEN U[t][i-1], U[t][i], and U[t][i+1] have been computed, THEN we
            // can compute U[t+1][i]
            for (std::size_t i = 0; i != nx; ++i)
            {
                next[i] = dataflow(
                        hpx::launch::async, Op,
                        current[idx(i, -1, nx)], current[i], current[idx(i, +1, nx)]
                    );
            }
        }

        // Now the asynchronous computation is running; the above for-loop does not
        // wait on anything. There is no implicit waiting at the end of each timestep;
        // the computation of each U[t][i] will begin when as soon as its dependencies
        // are ready and hardware is available.

        // Return the solution at time-step 'nt'.
        return hpx::when_all(U[nt % 2]);
    }
};
//]
///////////////////////////////////////////////////////////////////////////////
int hpx_main(boost::program_options::variables_map& vm)
{
    boost::uint64_t nx = vm["nx"].as<boost::uint64_t>();   // Number of grid points.
    boost::uint64_t nt = vm["nt"].as<boost::uint64_t>();   // Number of steps.

    if (vm.count("no-header"))
        header = false;

    // Create the stepper object
    stepper step;

    // Measure execution time.
    boost::uint64_t t = hpx::util::high_resolution_clock::now();

    // Execute nt time steps on nx grid points.
    hpx::future<stepper::space> result = step.do_work(nx, nt);

    stepper::space solution = result.get();
    hpx::wait_all(solution);

    boost::uint64_t elapsed = hpx::util::high_resolution_clock::now() - t;

    // Print the final solution
    if (vm.count("results"))
    {
        for (std::size_t i = 0; i != nx; ++i)
            std::cout << "U[" << i << "] = " << solution[i].get() << std::endl;
    }

    boost::uint64_t const os_thread_count = hpx::get_os_thread_count();
    print_time_results(os_thread_count, elapsed, nx, nt, header);

    return hpx::finalize();
}

int main(int argc, char* argv[])
{
    using namespace boost::program_options;

    options_description desc_commandline;
    desc_commandline.add_options()
        ("results", "print generated results (default: false)")
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
        ( "no-header", "do not print out the csv header row")
    ;

    // Initialize and run HPX
    return hpx::init(desc_commandline, argc, argv);
}

