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

#include <hpx/config.hpp>
#if !defined(HPX_COMPUTE_DEVICE_CODE)

#include "communicator.hpp"

#include <hpx/algorithm.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/include/async.hpp>
#include <hpx/include/components.hpp>
#include <hpx/include/lcos.hpp>
#include <hpx/include/util.hpp>
#include <hpx/modules/program_options.hpp>
#include <hpx/modules/resiliency.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <functional>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
double const pi = std::acos(-1.0);

///////////////////////////////////////////////////////////////////////////////
// Our partition data type
class partition_data
{
public:
    partition_data()
      : data_(0)
      , size_(0)
    {
    }

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
        {
            data_[k] = std::sin(2 * pi *
                ((0.0 + subdomain_width * subdomain_index + k) /
                    static_cast<double>(subdomain_width * subdomains)));
        }
    }

    // Move constructors
    partition_data(partition_data&& other)
      : data_(std::move(other.data_))
      , size_(other.size_)
    {
    }

    partition_data& operator=(partition_data&& other)
    {
        data_ = std::move(other.data_);
        size_ = other.size_;

        return *this;
    }

    // Copy constructor to send through the wire
    partition_data(partition_data const& other)
      : data_(other.data_)
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

private:
    // Allow transfer over the wire.
    friend class hpx::serialization::access;

    template <typename Archive>
    void serialize(Archive& ar, const unsigned int)
    {
        // clang-format off
        ar & data_;
        ar & size_;
        // clang-format on
    }
};

using stencil = std::vector<partition_data>;

// Setup HPX channel boilerplate
using communication_type = partition_data;

HPX_REGISTER_CHANNEL_DECLARATION(communication_type);
HPX_REGISTER_CHANNEL(communication_type, stencil_communication);

double stencil_operation(double left, double center, double right)
{
    return 0.5 * (0.75) * left + (0.75) * center - 0.5 * (0.25) * right;
}

partition_data stencil_update(std::size_t sti, partition_data const& center,
    partition_data const& left, partition_data const& right)
{
    const std::size_t size = center.size() - 1;
    partition_data workspace(size + 2 * sti + 1);

    std::copy(end(left) - sti - 1, end(left) - 1, &workspace[0]);
    std::copy(begin(center), end(center) - 1, &workspace[sti]);
    std::copy(begin(right), begin(right) + sti + 1, &workspace[size + sti]);

    for (std::size_t t = 0; t != sti; ++t)
    {
        for (std::size_t k = 0; k != size + 2 * sti - 1 - 2 * t; ++k)
            workspace[k] = stencil_operation(
                workspace[k], workspace[k + 1], workspace[k + 2]);
    }

    workspace.resize(size + 1);

    return workspace;
}

int hpx_main(hpx::program_options::variables_map& vm)
{
    std::size_t num_subdomains =
        vm["subdomains"].as<std::size_t>();    // Number of partitions.
    std::size_t subdomain_width =
        vm["subdomain-width"].as<std::size_t>();    // Number of grid points.
    std::size_t iterations =
        vm["iterations"].as<std::size_t>();    // Number of steps.
    std::size_t sti =
        vm["steps-per-iteration"]
            .as<std::size_t>();    // Number of time steps per iteration

    std::size_t num_localities = hpx::get_num_localities(hpx::launch::sync);
    std::size_t rank = hpx::get_locality_id();

    hpx::chrono::high_resolution_timer t_main;

    // Define the two grids
    std::array<stencil, 2> U;
    for (stencil& s : U)
        s.resize(num_subdomains);

    hpx::for_loop(hpx::execution::par, 0, num_subdomains,
        [&U, subdomain_width, num_subdomains](std::size_t i) {
            U[0][i] =
                partition_data(subdomain_width, double(i), num_subdomains);
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
        comm.set(communicator_type::right, U[0][num_subdomains - 1], 0);
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
                            partition_data&& left = gg.get();
                            next[0] = stencil_update(
                                sti, current[0], left, current[1]);
                            comm.set(communicator_type::left, next[0], t + 1);
                        });
        }

        if (comm.has_neighbor(communicator_type::right))
        {
            r = comm.get(communicator_type::right, t)
                    .then(hpx::launch::async,
                        [sti, num_subdomains, &current, &next, &comm, t](
                            hpx::future<partition_data>&& gg) {
                            partition_data&& right = gg.get();
                            next[num_subdomains - 1] =
                                stencil_update(sti, current[num_subdomains - 1],
                                    current[num_subdomains - 2], right);
                            comm.set(communicator_type::right,
                                next[num_subdomains - 1], t + 1);
                        });
        }

        std::vector<hpx::future<partition_data>> futures;
        futures.resize(num_subdomains - 2);

        for (std::size_t i = 1; i < num_subdomains - 1; ++i)
        {
            futures[i - 1] =
                hpx::async(stencil_update, sti, std::ref(current[i]),
                    std::ref(current[i - 1]), std::ref(current[i + 1]));
        }

        hpx::for_loop(hpx::execution::par, 1, num_subdomains - 1,
            [&next, &futures](
                std::size_t i) { next[i] = futures[i - 1].get(); });

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

    // clang-format off
    desc_commandline.add_options()
        ("subdomain-width", value<std::size_t>()->default_value(100),
            "Local x dimension (of each partition)")
        ("iterations", value<std::size_t>()->default_value(4),
            "Number of time steps")
        ("steps-per-iteration", value<std::size_t>()->default_value(4),
            "Number of time steps per iterations")
        ("subdomains", value<std::size_t>()->default_value(4),
            "Number of partitions")
    ;
    // clang-format on

    // Initialize and run HPX, this example requires to run hpx_main on all
    // localities
    std::vector<std::string> cfg = {
        "hpx.run_hpx_main!=1",
    };

    hpx::init_params params;
    params.desc_cmdline = desc_commandline;
    params.cfg = std::move(cfg);
    return hpx::init(argc, argv, params);
}

#endif
