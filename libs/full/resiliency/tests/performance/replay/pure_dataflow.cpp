//  Copyright (c) 2019 National Technology & Engineering Solutions of Sandia,
//                     LLC (NTESS).
//  Copyright (c) 2014 Hartmut Kaiser
//  Copyright (c) 2014 Patricia Grubel
//  Copyright (c) 2019 Nikunj Gupta
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// This is the fourth in a series of examples demonstrating the development of
// a fully distributed solver for a simple 1D heat distribution problem.
//
// This example builds on example three. It futurizes the code from that
// example. Compared to example two this code runs much more efficiently. It
// allows for changing the amount of work executed in one HPX thread which
// enables tuning the performance for the optimal grain size of the
// computation. This example is still fully local but demonstrates nice
// scalability on SMP machines.

#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>

#include <hpx/algorithm.hpp>
#include <hpx/include/lcos_local.hpp>
#include <hpx/modules/synchronization.hpp>
#include <boost/range/irange.hpp>

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <memory>
#include <utility>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
double const pi = std::acos(-1.0);

///////////////////////////////////////////////////////////////////////////////
// Our partition data type
struct partition_data
{
public:
    partition_data(std::size_t size)
      : data_(size)
      , size_(size)
    {
    }

    partition_data(std::size_t subdomain_width, double subdomain_index,
        std::size_t subdomains)
      : data_(subdomain_width + 1)
      , size_(subdomain_width + 1)
    {
        for (std::size_t k = 0; k != subdomain_width + 1; ++k)
            data_[k] = std::sin(2 * pi *
                ((0.0 + subdomain_width * subdomain_index + k) /
                    static_cast<double>(subdomain_width * subdomains)));
    }

    partition_data(partition_data&& other)
      : data_(std::move(other.data_))
      , size_(other.size_)
    {
    }

    double& operator[](std::size_t idx)
    {
        return data_[idx];
    }
    double operator[](std::size_t idx) const
    {
        return data_[idx];
    }

    std::size_t size() const
    {
        return size_;
    }

    void resize(std::size_t size)
    {
        data_.resize(size);
        size_ = size;
    }

private:
    std::vector<double> data_;
    std::size_t size_;
};

std::ostream& operator<<(std::ostream& os, partition_data const& c)
{
    os << "{";
    for (std::size_t i = 0; i != c.size() - 1; ++i)
    {
        if (i != 0)
            os << ", ";
        os << c[i];
    }
    os << "}";
    return os;
}

///////////////////////////////////////////////////////////////////////////////
struct stepper
{
    // Our data for one time step
    typedef hpx::shared_future<partition_data> partition;
    typedef std::vector<partition> space;

    // Our operator
    static double stencil(double left, double center, double right)
    {
        return 0.5 * (0.75) * left + (0.75) * center - 0.5 * (0.25) * right;
    }

    // The partitioned operator, it invokes the heat operator above on all
    // elements of a partition.
    static partition_data heat_part(partition_data const& left_input,
        partition_data const& center_input, partition_data const& right_input)
    {
        std::size_t const size = center_input.size() - 1;
        HPX_ASSERT(size >= 3);

        partition_data workspace(3 * size + 1);

        int j = 0;
        for (std::size_t i = 0; i != size - 2; ++i, ++j)
            workspace[j] =
                stencil(left_input[i], left_input[i + 1], left_input[i + 2]);

        workspace[j] = stencil(
            left_input[size - 2], left_input[size - 1], center_input[0]);
        workspace[j + 1] =
            stencil(left_input[size - 1], center_input[0], center_input[1]);
        j += 2;

        for (std::size_t i = 0; i != size - 2; ++i, ++j)
            workspace[j] = stencil(
                center_input[i], center_input[i + 1], center_input[i + 2]);

        workspace[j] = stencil(
            center_input[size - 2], center_input[size - 1], right_input[0]);
        workspace[j + 1] =
            stencil(center_input[size - 1], right_input[0], right_input[1]);
        j += 2;

        for (std::size_t i = 0; i != size - 1; ++i, ++j)
            workspace[j] =
                stencil(right_input[i], right_input[i + 1], right_input[i + 2]);

        for (std::size_t t = 1; t != size; ++t)
        {
            for (std::size_t k = 0; k != 3 * size - 1 - 2 * t; ++k)
                workspace[k] =
                    stencil(workspace[k], workspace[k + 1], workspace[k + 2]);
        }

        workspace.resize(size + 1);

        return workspace;
    }

    hpx::future<space> do_work(std::size_t subdomains,
        std::size_t subdomain_width, std::size_t iterations, std::uint64_t nd,
        hpx::lcos::local::sliding_semaphore& sem)
    {
        using hpx::dataflow;
        using hpx::util::unwrapping;

        // U[t][i] is the state of position i at time t.
        std::vector<space> U(2);
        for (space& s : U)
            s.resize(subdomains);

        std::size_t b = 0;
        auto range = boost::irange(b, subdomains);
        hpx::ranges::for_each(hpx::execution::par, range,
            [&U, subdomain_width, subdomains](std::size_t i) {
                U[0][i] = hpx::make_ready_future(
                    partition_data(subdomain_width, double(i), subdomains));
            });

        auto Op = unwrapping(&stepper::heat_part);

        // Actual time step loop
        for (std::size_t t = 0; t != iterations; ++t)
        {
            space const& current = U[t % 2];
            space& next = U[(t + 1) % 2];

            for (std::size_t i = 0; i != subdomains; ++i)
            {
                next[i] = dataflow(hpx::launch::async, Op,
                    current[(i - 1 + subdomains) % subdomains], current[i],
                    current[(i + 1) % subdomains]);
            }

            // every nd time steps, attach additional continuation which will
            // trigger the semaphore once computation has reached this point
            if ((t % nd) == 0)
            {
                next[0].then([&sem, t](partition&&) {
                    // inform semaphore about new lower limit
                    sem.signal(t);
                });
            }

            // suspend if the tree has become too deep, the continuation above
            // will resume this thread once the computation has caught up
            sem.wait(t);
        }

        // Return the solution at time-step 'iterations'.
        return hpx::when_all(U[iterations % 2]);
    }
};

///////////////////////////////////////////////////////////////////////////////
int hpx_main(hpx::program_options::variables_map& vm)
{
    std::uint64_t subdomains =
        vm["subdomains"].as<std::uint64_t>();    // Number of partitions.
    std::uint64_t subdomain_width =
        vm["subdomain-width"].as<std::uint64_t>();    // Number of grid points.
    std::uint64_t iterations =
        vm["iterations"].as<std::uint64_t>();    // Number of steps.
    std::uint64_t nd =
        vm["nd"].as<std::uint64_t>();    // Max depth of dep tree.

    // Create the stepper object
    stepper step;

    std::cout << "Starting 1d stencil with dataflow" << std::endl;

    // Measure execution time.
    std::uint64_t t = hpx::chrono::high_resolution_clock::now();

    {
        // limit depth of dependency tree
        hpx::lcos::local::sliding_semaphore sem(nd);

        hpx::future<stepper::space> result =
            step.do_work(subdomains, subdomain_width, iterations, nd, sem);

        stepper::space solution = result.get();
        hpx::wait_all(solution);
    }

    std::cout << "Time elapsed: "
              << static_cast<double>(
                     hpx::chrono::high_resolution_clock::now() - t) /
            1e9
              << std::endl;
    std::cout << "Errors occurred: 0" << std::endl;

    // for (std::size_t i = 0; i != subdomains; ++i)
    //     std::cout << solution[i].get() << " ";
    // std::cout << std::endl;

    return hpx::finalize();
}

int main(int argc, char* argv[])
{
    using namespace hpx::program_options;

    // Configure application-specific options.
    options_description desc_commandline;

    desc_commandline.add_options()(
        "results", "print generated results (default: false)");

    desc_commandline.add_options()("subdomain-width",
        value<std::uint64_t>()->default_value(128),
        "Local x dimension (of each partition)");

    desc_commandline.add_options()("iterations",
        value<std::uint64_t>()->default_value(10), "Number of time steps");

    desc_commandline.add_options()("nd",
        value<std::uint64_t>()->default_value(10),
        "Number of time steps to allow the dependency tree to grow to");

    desc_commandline.add_options()("subdomains",
        value<std::uint64_t>()->default_value(10), "Number of partitions");

    // Initialize and run HPX
    hpx::init_params init_args;
    init_args.desc_cmdline = desc_commandline;

    return hpx::init(argc, argv, init_args);
}
