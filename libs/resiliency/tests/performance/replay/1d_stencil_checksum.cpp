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
#include <hpx/include/parallel_algorithm.hpp>
#include <hpx/modules/resiliency.hpp>
#include <hpx/modules/synchronization.hpp>
#include <boost/range/irange.hpp>

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

struct validate_exception : std::exception
{
};

///////////////////////////////////////////////////////////////////////////////
double const pi = std::acos(-1.0);
double const checksum_tol = 1.0e-10;

// Variable to count the number of failed attempts
std::atomic<int> counter(0);

// Variables to generate errors
std::random_device rd;
std::mt19937 gen(rd());

///////////////////////////////////////////////////////////////////////////////
// Our partition data type
struct partition_data
{
public:
    partition_data(std::size_t size)
      : data_(size)
      , size_(size)
      , checksum_(0.0)
      , test_value_(0.0)
    {
    }

    partition_data(std::size_t subdomain_width, double subdomain_index,
        std::size_t subdomains)
      : data_(subdomain_width + 1)
      , size_(subdomain_width + 1)
      , test_value_(0.0)
    {
        checksum_ = 0.0;
        for (std::size_t k = 0; k != subdomain_width + 1; ++k)
        {
            data_[k] = std::sin(2 * pi *
                ((0.0 + subdomain_width * subdomain_index + k) /
                    static_cast<double>(subdomain_width * subdomains)));
            checksum_ += data_[k];
        }
    }

    partition_data(partition_data&& other)
      : data_(std::move(other.data_))
      , size_(other.size_)
      , checksum_(other.checksum_)
      , test_value_(other.test_value_)
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
    double checksum() const
    {
        return checksum_;
    }
    void set_checksum()
    {
        checksum_ = std::accumulate(data_.begin(), data_.end(), 0.0);
    }
    void set_test_value(double test_value)
    {
        test_value_ = test_value;
    }
    double verify_result() const
    {
        return std::abs(checksum_ - test_value_);
    }

    friend std::vector<double>::const_iterator begin(const partition_data& v)
    {
        return begin(v.data_);
    }
    friend std::vector<double>::const_iterator end(const partition_data& v)
    {
        return end(v.data_);
    }

    void resize(std::size_t size)
    {
        data_.resize(size);
        size_ = size;
    }

private:
    std::vector<double> data_;
    std::size_t size_;
    double checksum_;
    double test_value_;
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

// Function declaration
bool validate_result(partition_data const& f);

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

    static double left_flux(double left, double center)
    {
        return (0.625) * left - (0.125) * center;
    }

    static double right_flux(double center, double right)
    {
        return 0.5 * (0.75) * center + (1.125) * right;
    }

    // The partitioned operator, it invokes the heat operator above on all
    // elements of a partition.
    static partition_data heat_part(std::size_t sti, double error,
        partition_data const& left_input, partition_data const& center_input,
        partition_data const& right_input)
    {
        static thread_local std::exponential_distribution<> dist_(error);

        double num = dist_(gen);
        bool error_flag = false;

        // Probability of error occurrence is proportional to exp(-error_rate)
        if (num > 1.0)
        {
            error_flag = true;
            ++counter;
        }

        const std::size_t size = center_input.size() - 1;
        partition_data workspace(size + 2 * sti + 1);

        std::copy(
            end(left_input) - sti - 1, end(left_input) - 1, &workspace[0]);
        std::copy(begin(center_input), end(center_input) - 1, &workspace[sti]);
        std::copy(begin(right_input), begin(right_input) + sti + 1,
            &workspace[size + sti]);

        double left_checksum =
            std::accumulate(end(left_input) - sti - 1, end(left_input), 0.0);
        double right_checksum = std::accumulate(
            begin(right_input), begin(right_input) + sti + 1, 0.0);

        double checksum = left_checksum - center_input[0] +
            center_input.checksum() - right_input[0] + right_checksum;

        for (std::size_t t = 0; t != sti; ++t)
        {
            checksum -= left_flux(workspace[0], workspace[1]);
            checksum -= right_flux(workspace[size + 2 * sti - 1 - 2 * t],
                workspace[size + 2 * sti - 2 * t]);
            for (std::size_t k = 0; k != size + 2 * sti - 1 - 2 * t; ++k)
                workspace[k] =
                    stencil(workspace[k], workspace[k + 1], workspace[k + 2]);
        }

        workspace.resize(size + 1);
        workspace.set_checksum();
        workspace.set_test_value(checksum);

        // Artificial error injection to get replay in action
        if (error_flag)
            throw validate_exception();

        return workspace;
    }

    hpx::future<space> do_work(std::size_t subdomains,
        std::size_t subdomain_width, std::size_t iterations, std::size_t sti,
        std::uint64_t nd, std::uint64_t n_value, double error,
        hpx::lcos::local::sliding_semaphore& sem)
    {
        using hpx::resiliency::experimental::dataflow_replay_validate;
        using hpx::util::unwrapping;

        // U[t][i] is the state of position i at time t.
        std::vector<space> U(2);
        for (space& s : U)
            s.resize(subdomains);

        std::size_t b = 0;
        auto range = boost::irange(b, subdomains);
        using hpx::parallel::execution::par;
        hpx::parallel::for_each(par, std::begin(range), std::end(range),
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
                // explicitly unwrap future
                hpx::future<partition_data> f =
                    dataflow_replay_validate(n_value, validate_result, Op, sti,
                        error, current[(i - 1 + subdomains) % subdomains],
                        current[i], current[(i + 1) % subdomains]);
                next[i] = std::move(f);
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

bool validate_result(partition_data const& f)
{
    if (f.verify_result() <= checksum_tol)
        return true;

    return false;
}

///////////////////////////////////////////////////////////////////////////////
int hpx_main(hpx::program_options::variables_map& vm)
{
    std::uint64_t n_value =
        vm["n-value"].as<std::uint64_t>();    // Number of partitions.
    std::uint64_t subdomains =
        vm["subdomains"].as<std::uint64_t>();    // Number of partitions.
    std::uint64_t subdomain_width =
        vm["subdomain-width"].as<std::uint64_t>();    // Number of grid points.
    std::uint64_t iterations =
        vm["iterations"].as<std::uint64_t>();    // Number of steps.
    std::uint64_t nd =
        vm["nd"].as<std::uint64_t>();    // Max depth of dep tree.
    std::uint64_t sti =
        vm["steps-per-iteration"]
            .as<std::uint64_t>();    // Number of time steps per iteration
    double error = vm["error-rate"].as<double>();

    // Create the stepper object
    stepper step;

    std::cout << "Starting 1d stencil with dataflow replay validate"
              << std::endl;

    // Measure execution time.
    std::uint64_t t = hpx::util::high_resolution_clock::now();

    {
        // limit depth of dependency tree
        hpx::lcos::local::sliding_semaphore sem(nd);

        hpx::future<stepper::space> result = step.do_work(subdomains,
            subdomain_width, iterations, sti, nd, n_value, error, sem);

        stepper::space solution = result.get();
        hpx::wait_all(solution);
    }

    std::cout << "Time elapsed: "
              << static_cast<double>(
                     hpx::util::high_resolution_clock::now() - t) /
            1e9
              << std::endl;
    std::cout << "Errors occurred: " << counter << std::endl;

    return hpx::finalize();
}

int main(int argc, char* argv[])
{
    using namespace hpx::program_options;

    // Configure application-specific options.
    options_description desc_commandline;

    desc_commandline.add_options()(
        "results", "print generated results (default: false)");

    desc_commandline.add_options()("n-value",
        value<std::uint64_t>()->default_value(5), "Number of allowed replays");

    desc_commandline.add_options()("error-rate",
        value<double>()->default_value(5), "Error rate for injecting errors");

    desc_commandline.add_options()("subdomain-width",
        value<std::uint64_t>()->default_value(128),
        "Local x dimension (of each partition)");

    desc_commandline.add_options()("iterations",
        value<std::uint64_t>()->default_value(10), "Number of time steps");

    desc_commandline.add_options()("steps-per-iteration",
        value<std::uint64_t>()->default_value(16),
        "Number of time steps per iterations");

    desc_commandline.add_options()("nd",
        value<std::uint64_t>()->default_value(10),
        "Number of time steps to allow the dependency tree to grow to");

    desc_commandline.add_options()("subdomains",
        value<std::uint64_t>()->default_value(10), "Number of partitions");

    // Initialize and run HPX
    return hpx::init(desc_commandline, argc, argv);
}
