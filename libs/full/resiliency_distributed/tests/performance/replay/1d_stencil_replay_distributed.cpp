// Copyright (c) 2020 Nikunj Gupta
//
// SPDX-License-Identifier: BSL-1.0
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// This is the a separate examples demonstrating the development
// of a fully distributed solver for a simple 1D heat distribution problem.
//
// This example makes use of LCOS channels to send and receive
// elements.

#include "communicator.hpp"

#include <hpx/algorithm.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/include/async.hpp>
#include <hpx/include/components.hpp>
#include <hpx/include/compute.hpp>
#include <hpx/include/lcos.hpp>
#include <hpx/include/util.hpp>
#include <hpx/modules/resiliency.hpp>
#include <hpx/program_options/options_description.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <iostream>
#include <string>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
double const pi = std::acos(-1.0);
double const checksum_tol = 1.0e-10;

// Random number generator
std::random_device randomizer;
std::mt19937 gen(randomizer());
std::uniform_int_distribution<std::size_t> dis(1, 100);
///////////////////////////////////////////////////////////////////////////////
// Our partition data type
class partition_data
{
public:
    partition_data()
      : data_(0)
      , size_(0)
      , checksum_(0.0)
      , test_value_(0.0)
    {
    }

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

    // Move constructors
    partition_data(partition_data&& other)
      : data_(std::move(other.data_))
      , size_(other.size_)
      , checksum_(other.checksum_)
      , test_value_(other.test_value_)
    {
    }

    partition_data& operator=(partition_data&& other)
    {
        data_ = std::move(other.data_);
        size_ = other.size_;
        checksum_ = other.checksum_;
        test_value_ = other.test_value_;

        return *this;
    }

    // Copy constructor to send through the wire
    // Copy constructor to send through the wire
    partition_data(partition_data const& other)
      : data_(other.data_)
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

private:
    // Allow transfer over the wire.
    friend class hpx::serialization::access;

    template <typename Archive>
    void serialize(Archive& ar, const unsigned int version)
    {
        ar& data_;
        ar& size_;
        ar& checksum_;
        ar& test_value_;
    }
};

using stencil = std::vector<partition_data>;

// Setup HPX channel boilerplate
using communication_type = partition_data;

HPX_REGISTER_CHANNEL_DECLARATION(communication_type);
HPX_REGISTER_CHANNEL(communication_type, stencil_communication);

double left_flux(double left, double center)
{
    return (0.625) * left - (0.125) * center;
}

double right_flux(double center, double right)
{
    return 0.5 * (0.75) * center + (1.125) * right;
}

double stencil_operation(double left, double center, double right)
{
    return 0.5 * (0.75) * left + (0.75) * center - 0.5 * (0.25) * right;
}

partition_data stencil_update(std::size_t sti, partition_data const& center,
    partition_data const& left, partition_data const& right, std::size_t error,
    bool is_faulty_node)
{
    const std::size_t size = center.size() - 1;
    partition_data workspace(size + 2 * sti + 1);

    std::copy(end(left) - sti - 1, end(left) - 1, &workspace[0]);
    std::copy(begin(center), end(center) - 1, &workspace[sti]);
    std::copy(begin(right), begin(right) + sti + 1, &workspace[size + sti]);

    double left_checksum = std::accumulate(end(left) - sti - 1, end(left), 0.0);
    double right_checksum =
        std::accumulate(begin(right), begin(right) + sti + 1, 0.0);

    double checksum = left_checksum - center[0] + center.checksum() - right[0] +
        right_checksum;

    for (std::size_t t = 0; t != sti; ++t)
    {
        checksum -= left_flux(workspace[0], workspace[1]);
        checksum -= right_flux(workspace[size + 2 * sti - 1 - 2 * t],
            workspace[size + 2 * sti - 2 * t]);
        for (std::size_t k = 0; k != size + 2 * sti - 1 - 2 * t; ++k)
            workspace[k] = stencil_operation(
                workspace[k], workspace[k + 1], workspace[k + 2]);
    }

    workspace.resize(size + 1);
    workspace.set_checksum();
    workspace.set_test_value(checksum);

    if (is_faulty_node)
    {
        if (dis(gen) < (error * 10))
            throw std::runtime_error("Error occured");
    }
    else
    {
        if (dis(gen) < error)
            throw std::runtime_error("Error occured");
    }

    return workspace;
}

bool validate_result(partition_data const& f)
{
    if (f.verify_result() <= checksum_tol)
        return true;

    return false;
}

partition_data stencil_update_distributed(std::size_t sti,
    partition_data center, partition_data left, partition_data right,
    std::size_t error, bool is_faulty_node)
{
    return hpx::resiliency::experimental::async_replay(
        100, stencil_update, sti, center, left, right, error, is_faulty_node)
        .get();
}
HPX_PLAIN_ACTION(stencil_update_distributed, stencil_action);

int hpx_main(boost::program_options::variables_map& vm)
{
    std::size_t n_value =
        vm["n-value"].as<std::size_t>();    // Number of partitions.
    std::size_t num_subdomains =
        vm["subdomains"].as<std::size_t>();    // Number of partitions.
    std::size_t subdomain_width =
        vm["subdomain-width"].as<std::size_t>();    // Number of grid points.
    std::size_t iterations =
        vm["iterations"].as<std::size_t>();    // Number of steps.
    std::size_t sti =
        vm["steps-per-iteration"]
            .as<std::size_t>();    // Number of time steps per iteration
    std::size_t faults = vm["faults"].as<std::size_t>();
    std::size_t errors = vm["errors"].as<std::size_t>();

    std::size_t num_localities = hpx::get_num_localities(hpx::launch::sync);
    std::size_t rank = hpx::get_locality_id();

    bool is_faulty_node = false;
    if (faults != 0)
    {
        for (std::size_t i = 0; i < faults; ++i)
        {
            if (rank == i * (num_localities / faults))
                is_faulty_node = true;
        }
    }

    std::size_t counter = 0;
    std::vector<std::vector<hpx::id_type>> locales;

    if (faults != 0)
    {
        locales.reserve((num_localities / faults));

        for (std::size_t i = 1; i < (num_localities / faults); ++i)
        {
            hpx::id_type id_1 = hpx::naming::get_id_from_locality_id(
                (rank + i) % num_localities);

            std::vector<hpx::id_type> local{id_1};
            locales.push_back(local);
        }
    }

    hpx::util::high_resolution_timer t_main;

    // Define the two grids
    std::array<stencil, 2> U;
    for (stencil& s : U)
        s.resize(num_subdomains);

    std::size_t b = 0;
    auto range = boost::irange(b, num_subdomains);
    hpx::ranges::for_each(hpx::parallel::execution::par, range,
        [&U, subdomain_width, num_subdomains](std::size_t i) {
            U[0][i] = std::move(
                partition_data(subdomain_width, double(i), num_subdomains));
        });

    // Setup communicator
    using communicator_type = communicator<partition_data>;
    communicator_type comm(rank, num_localities);

    if (rank == 0)
    {
        std::cout << "Starting benchmark with " << num_localities << std::endl;
    }

    if (comm.has_neighbor(communicator_type::left))
    {
        // Send initial value to the left neighbor
        comm.set(communicator_type::left, U[0][0], 0);
    }
    if (comm.has_neighbor(communicator_type::right))
    {
        // Send initial value to the right neighbor
        comm.set(communicator_type::right, U[0][0], 0);
    }

    for (std::size_t t = 0; t < iterations; ++t)
    {
        stencil const& current = U[t % 2];
        stencil& next = U[(t + 1) % 2];

        hpx::future<void> l = hpx::make_ready_future();
        hpx::future<void> r = hpx::make_ready_future();

        if (comm.has_neighbor(communicator_type::left))
        {
            l = comm.get(communicator_type::left, t)
                    .then(hpx::launch::async,
                        [sti, &current, &next, &comm, t](
                            hpx::future<partition_data>&& gg) {
                            partition_data left = std::move(gg.get());
                            next[0] = stencil_update(
                                sti, current[0], left, current[1], 0, false);
                            comm.set(communicator_type::left, next[0], t + 1);
                        });
        }

        if (comm.has_neighbor(communicator_type::right))
        {
            r = comm.get(communicator_type::right, t)
                    .then(hpx::launch::async,
                        [sti, num_subdomains, &current, &next, &comm, t](
                            hpx::future<partition_data>&& gg) {
                            partition_data right = std::move(gg.get());
                            next[num_subdomains - 1] = stencil_update(sti,
                                current[num_subdomains - 1],
                                current[num_subdomains - 2], right, 0, false);
                            comm.set(communicator_type::right,
                                next[num_subdomains - 1], t + 1);
                        });
        }

        std::vector<hpx::future<partition_data>> futures;
        futures.resize(num_subdomains - 2);

        for (std::size_t i = 1; i < num_subdomains - 1; ++i)
        {
            futures[i - 1] =
                hpx::resiliency::experimental::async_replay_validate(n_value,
                    validate_result, stencil_update, sti, current[i],
                    current[i - 1], current[i + 1], errors, is_faulty_node)
                    .then(hpx::launch::async,
                        [sti, &current, errors, i, is_faulty_node, &locales,
                            &counter, num_localities,
                            faults](hpx::future<partition_data>&& gg) {
                            if (gg.has_exception())
                            {
                                if (faults != 0)
                                    counter = (counter + 1) %
                                        (num_localities / faults);
                                stencil_action ac;
                                return hpx::resiliency::experimental::
                                    async_replay_validate(locales[counter],
                                        validate_result, ac, sti, current[i],
                                        current[i - 1], current[i + 1], errors,
                                        false)
                                        .get();
                            }
                            else
                                return gg.get();
                        });
        }

        b = 1;
        auto range = boost::irange(b, num_subdomains - 1);
        hpx::ranges::for_each(hpx::parallel::execution::par, range,
            [&next, &futures](
                std::size_t i) { next[i] = std::move(futures[i - 1].get()); });

        hpx::wait_all(l, r);
    }

    double telapsed = t_main.elapsed();

    if (rank == 0)
        std::cout << "Total time: " << telapsed << std::endl;

    return hpx::finalize();
}

int main(int argc, char* argv[])
{
    using namespace hpx::program_options;

    // Configure application-specific options.
    options_description desc_commandline;

    desc_commandline.add_options()("n-value",
        value<std::size_t>()->default_value(5), "Number of allowed replays");

    desc_commandline.add_options()("errors",
        value<std::size_t>()->default_value(2), "Number of faulty nodes");

    desc_commandline.add_options()("faults",
        value<std::size_t>()->default_value(1), "Number of faulty nodes");

    desc_commandline.add_options()("subdomain-width",
        value<std::size_t>()->default_value(8000),
        "Local x dimension (of each partition)");

    desc_commandline.add_options()("iterations",
        value<std::size_t>()->default_value(256), "Number of time steps");

    desc_commandline.add_options()("steps-per-iteration",
        value<std::size_t>()->default_value(512),
        "Number of time steps per iterations");

    desc_commandline.add_options()("subdomains",
        value<std::size_t>()->default_value(384), "Number of partitions");

    // Initialize and run HPX, this example requires to run hpx_main on all
    // localities
    std::vector<std::string> const cfg = {
        "hpx.run_hpx_main!=1",
    };

    return hpx::init(desc_commandline, argc, argv, cfg);
}
