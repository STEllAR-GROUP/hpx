//  Copyright (c) 2025 Lukas Zeil
//  Copyright (c) 2025 Alexander Strack
//  Copyright (c) 2025 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>

#if !defined(HPX_COMPUTE_DEVICE_CODE)
#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/modules/collectives.hpp>
#include <hpx/modules/testing.hpp>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

using namespace hpx::collectives;

constexpr char const* scatter_direct_basename = "/test/scatter_direct/";
constexpr char const* reduce_direct_basename = "/test/reduce_direct/";
constexpr char const* broadcast_direct_basename = "/test/broadcast_direct/";
constexpr char const* gather_direct_basename = "/test/gather_direct/";

struct vector_adder
{
    std::vector<int> operator()(
        std::vector<int> const& a, std::vector<int> const& b) const
    {
        if (a.size() != b.size())
        {
            throw std::runtime_error("Vector sizes must match!");
        }

        std::vector<int> result(a.size());
        for (size_t i = 0; i < a.size(); ++i)
        {
            result[i] = a[i] + b[i];
        }
        return result;
    }
};

std::vector<std::vector<int>> generate_matrix(
    int localities, int test_size, int start_val)
{
    std::vector<std::vector<int>> result;
    for (int i = 0; i < localities; ++i)
    {
        result.push_back(std::vector<int>(test_size, start_val + i));
    }
    return result;
}

void create_parent_dir(std::filesystem::path const& file_path)
{
    // Create parent directory if does not exist
    std::filesystem::path dir_path = file_path.parent_path();
    if (!std::filesystem::exists(dir_path))
    {
        if (std::filesystem::create_directories(dir_path))
        {
            std::cout << "Directory created: " << dir_path << "\n";
        }
        else
        {
            throw std::runtime_error(
                "Failed to create directory: " + dir_path.string());
        }
    }
}

std::pair<double, double> compute_moments(std::vector<double> const& data)
{
    // Compute mean
    double sum = 0.0;
    for (double x : data)
    {
        sum += x;
    }
    double mean = sum / static_cast<double>(data.size());

    // Compute variance (population variance)
    double varianceSum = 0.0;
    for (double x : data)
    {
        varianceSum += (x - mean) * (x - mean);
    }
    double variance = varianceSum / static_cast<double>(data.size());

    return std::make_pair(mean, variance);
}

void write_to_file(std::string const& collective, std::string const& type,
    int arity, std::size_t num_l, int lpn, int size, std::size_t iterations,
    std::vector<double> const& result)
{
    // Compute mean and variance
    auto moments = compute_moments(result);
    // Compute nodes
    std::size_t nodes = num_l / static_cast<std::size_t>(lpn);
    auto threads = hpx::get_os_thread_count();
    // Print info
    std::string msg = "\nCollective:        {1}\n"
                      "Type:              {2}\n"
                      "Arity:             {3}\n"
                      "Nodes:             {4}\n"
                      "Localities:        {5}\n"
                      "Localities/Node:   {6}\n"
                      "HPX threads:       {7}\n"
                      "Size/Locality:     {8}\n"
                      "Iterations:        {9}\n"
                      "Mean runtime:      {10}\n"
                      "Variance:          {11}\n";
    hpx::util::format_to(std::cout, msg, collective, type, nodes, num_l, lpn,
        threads, size, iterations, moments.first, moments.second)
        << std::flush;

    // Create directory
    std::string runtime_file_path =
        "result/hpx/" + collective + "/runtimes_" + collective + "_" + type;
    if (arity != -1 && type == "hierarchical")
    {
        runtime_file_path = runtime_file_path + "_" + std::to_string(arity);
    }
    create_parent_dir(runtime_file_path);

    // Add header if necessary
    std::string const header = "collective;type;arity;nodes;localities;lpn;"
                               "threads;size;iterations;mean;variance;\n";
    // Read existing content
    std::ifstream infile(runtime_file_path);
    std::stringstream buffer;
    buffer << infile.rdbuf();
    std::string contents = buffer.str();
    infile.close();
    // Only append if header not present
    if (contents.find(header) == std::string::npos)
    {
        std::ofstream outfile(runtime_file_path, std::ios_base::app);
        outfile << header;
        outfile.close();
    }

    // Add runtimes
    std::ofstream outfile;
    outfile.open(runtime_file_path, std::ios_base::app);
    outfile << collective << ";" << type << ";" << arity << ";" << nodes << ";"
            << num_l << ";" << lpn << ";" << threads << ";" << size << ";"
            << iterations << ";" << moments.first << ";" << moments.second
            << ";\n";
    outfile.close();
}

////////////////////////////////////////////////////////////////////////////////////////
// Hierarchical collectives
void test_scatter_hierarchical(int arity, int lpn, std::size_t iterations,
    int test_size, std::string const& operation)
{
    // Get parameters
    std::size_t const num_localities =
        hpx::get_num_localities(hpx::launch::sync);
    std::size_t const this_locality = hpx::get_locality_id();
    // Ensure at least two localities
    HPX_TEST_LTE(static_cast<std::size_t>(2), num_localities);
    // Create hierchical communicators
    auto communicators =
        create_hierarchical_communicator(scatter_direct_basename,
            num_sites_arg(num_localities), this_site_arg(this_locality),
            arity_arg(arity), generation_arg(1), root_site_arg(0));
    // Barrier for synchronization
    char const* const barrier_test_name = "/test/barrier/hierarchical";
    hpx::distributed::barrier barrier(barrier_test_name);
    // Result vector
    std::vector<double> result(iterations, 0.0);
    // Data
    std::vector<std::vector<int>> send_data;
    std::vector<int> recv_data;
    hpx::future<std::vector<int>> ft_data;

    for (std::size_t i = 0; i != iterations; ++i)
    {
        if (this_locality == 0)
        {
            send_data = generate_matrix(static_cast<int>(num_localities),
                test_size, static_cast<int>(42 + i));
        }

        // Time collective
        hpx::chrono::high_resolution_timer const timer;
        if (this_locality == 0)
        {
            // NOLINTNEXTLINE(bugprone-use-after-move)
            ft_data = scatter_to(communicators, std::move(send_data),
                this_site_arg(this_locality), generation_arg(i + 1));
        }
        else
        {
            ft_data = scatter_from<std::vector<int>>(communicators,
                this_site_arg(this_locality), generation_arg(i + 1));
        }
        recv_data = ft_data.get();
        // Synchronize
        barrier.wait();
        // Write runtime into vector
        result[i] = timer.elapsed();

        // Check for correctness
        for (std::size_t check : recv_data)
        {
            HPX_TEST_EQ(42 + i + this_locality, check);
        }
    }

    if (this_locality == 0)
    {
        write_to_file(operation, "hierarchical", arity, num_localities, lpn,
            test_size, iterations, result);
    }
}

void test_reduce_hierarchical(int arity, int lpn, std::size_t iterations,
    int test_size, std::string const& operation)
{
    // Get parameters
    std::size_t const num_localities =
        hpx::get_num_localities(hpx::launch::sync);
    std::size_t const this_locality = hpx::get_locality_id();
    // Ensure at least two localities
    HPX_TEST_LTE(static_cast<std::size_t>(2), num_localities);
    // Create communicator
    auto communicators =
        create_hierarchical_communicator(reduce_direct_basename,
            num_sites_arg(num_localities), this_site_arg(this_locality),
            arity_arg(arity), generation_arg(1), root_site_arg(0));
    // Barrier for synchronization
    char const* const barrier_test_name = "/test/barrier/hierarchical";
    hpx::distributed::barrier barrier(barrier_test_name);
    // Result vector
    std::vector<double> result(iterations, 0.0);
    // Data
    std::vector<int> send_data;
    std::vector<int> recv_data;
    hpx::future<std::vector<int>> ft_data;

    for (std::size_t i = 0; i != iterations; ++i)
    {
        send_data = std::vector<int>(test_size, static_cast<int>(i));

        // Time collective
        hpx::chrono::high_resolution_timer const timer;
        if (this_locality == 0)
        {
            ft_data =
                // NOLINTNEXTLINE(bugprone-use-after-move)
                reduce_here(communicators, std::move(send_data), vector_adder{},
                    this_site_arg(this_locality), generation_arg(i + 1));
            recv_data = ft_data.get();
        }
        else
        {
            hpx::future<void> finished = reduce_there(communicators,
                // NOLINTNEXTLINE(bugprone-use-after-move)
                std::move(send_data), vector_adder{},
                this_site_arg(this_locality), generation_arg(i + 1));
            finished.get();
        }
        // Synchronize
        barrier.wait();
        // Write runtime into vector
        result[i] = timer.elapsed();

        // Check for correctness
        if (this_locality == 0)
        {
            HPX_TEST_EQ(
                i * num_localities, static_cast<std::size_t>(recv_data[0]));
        }
    }

    if (this_locality == 0)
    {
        write_to_file(operation, "hierarchical", arity, num_localities, lpn,
            test_size, iterations, result);
    }
}

void test_broadcast_hierarchical(int arity, int lpn, std::size_t iterations,
    int test_size, std::string const& operation)
{
    // Get parameters
    std::size_t const num_localities =
        hpx::get_num_localities(hpx::launch::sync);
    std::size_t const this_locality = hpx::get_locality_id();
    // Ensure at least two localities
    HPX_TEST_LTE(static_cast<std::size_t>(2), num_localities);
    // Create communicator
    auto communicators =
        create_hierarchical_communicator(broadcast_direct_basename,
            num_sites_arg(num_localities), this_site_arg(this_locality),
            arity_arg(arity), generation_arg(1), root_site_arg(0));
    // Barrier for synchronization
    char const* const barrier_test_name = "/test/barrier/hierarchical";
    hpx::distributed::barrier barrier(barrier_test_name);
    // Result vector
    std::vector<double> result(iterations, 0.0);
    // Data
    std::vector<int> send_data;
    std::vector<int> recv_data;
    hpx::future<std::vector<int>> ft_data;

    for (std::size_t i = 0; i != iterations; ++i)
    {
        if (this_locality == 0)
        {
            send_data = std::vector<int>(test_size, static_cast<int>(i));
        }

        // Time collective
        hpx::chrono::high_resolution_timer const timer;
        if (this_locality == 0)
        {
            // NOLINTNEXTLINE(bugprone-use-after-move)
            ft_data = broadcast_to(communicators, std::move(send_data),
                this_site_arg(this_locality), generation_arg(i + 1));
        }
        else
        {
            ft_data = broadcast_from<std::vector<int>>(communicators,
                this_site_arg(this_locality), generation_arg(i + 1));
        }
        recv_data = ft_data.get();
        // Synchronize
        barrier.wait();
        // Write runtime into vector
        result[i] = timer.elapsed();

        // Check for correctness
        if (this_locality == 0)
        {
            HPX_TEST_EQ(i, static_cast<std::size_t>(recv_data[0]));
        }
    }

    if (this_locality == 0)
    {
        write_to_file(operation, "hierarchical", arity, num_localities, lpn,
            test_size, iterations, result);
    }
}

void test_gather_hierarchical(int arity, int lpn, std::size_t iterations,
    int test_size, std::string const& operation)
{
    // Get parameters
    std::size_t const num_localities =
        hpx::get_num_localities(hpx::launch::sync);
    std::size_t const this_locality = hpx::get_locality_id();
    // Ensure at least two localities
    HPX_TEST_LTE(static_cast<std::size_t>(2), num_localities);
    // Create communicator
    auto communicators =
        create_hierarchical_communicator(gather_direct_basename,
            num_sites_arg(num_localities), this_site_arg(this_locality),
            arity_arg(arity), generation_arg(1), root_site_arg(0));
    // Barrier for synchronization
    char const* const barrier_test_name = "/test/barrier/hierarchical";
    hpx::distributed::barrier barrier(barrier_test_name);
    // Result vector
    std::vector<double> result(iterations, 0.0);
    // Data
    std::vector<int> send_data;
    std::vector<std::vector<int>> recv_data;
    hpx::future<std::vector<std::vector<int>>> ft_data;

    for (std::size_t i = 0; i != iterations; ++i)
    {
        send_data =
            std::vector<int>(test_size, static_cast<int>(i + this_locality));

        // Time collective
        hpx::chrono::high_resolution_timer const timer;
        if (this_locality == 0)
        {
            // NOLINTNEXTLINE(bugprone-use-after-move)
            ft_data = gather_here(communicators, std::move(send_data),
                this_site_arg(this_locality), generation_arg(i + 1));
            recv_data = ft_data.get();
        }
        else
        {
            hpx::future<void> finished =
                // NOLINTNEXTLINE(bugprone-use-after-move)
                gather_there(communicators, std::move(send_data),
                    this_site_arg(this_locality), generation_arg(i + 1));
            finished.get();
        }
        // Synchronize
        barrier.wait();
        // Write runtime into vector
        result[i] = timer.elapsed();

        // Check for correctness
        if (this_locality == 0)
        {
            for (int j = 0; j < static_cast<int>(num_localities); ++j)
            {
                HPX_TEST_EQ(i + j, static_cast<std::size_t>(recv_data[j][0]));
            }
        }
    }

    if (this_locality == 0)
    {
        write_to_file(operation, "hierarchical", arity, num_localities, lpn,
            test_size, iterations, result);
    }
}

////////////////////////////////////////////////////////////////////////////////////////
// One shot collectives
void test_one_shot_use_scatter(int lpn, std::size_t iterations, int test_size,
    std::string const& operation)
{
    // Get parameters
    std::size_t const num_localities =
        hpx::get_num_localities(hpx::launch::sync);
    std::size_t const this_locality = hpx::get_locality_id();
    // Ensure at least two localities
    HPX_TEST_LTE(static_cast<std::size_t>(2), num_localities);
    // Barrier for synchronization
    char const* const barrier_test_name = "/test/barrier/single";
    hpx::distributed::barrier barrier(barrier_test_name);
    // Result vector
    std::vector<double> result(iterations, 0.0);
    // Data
    std::vector<std::vector<int>> send_data;
    std::vector<int> recv_data;
    hpx::future<std::vector<int>> ft_data;

    for (std::size_t i = 0; i != iterations; ++i)
    {
        if (this_locality == 0)
        {
            send_data = generate_matrix(static_cast<int>(num_localities),
                test_size, static_cast<int>(42 + i));
        }

        // Time collective
        hpx::chrono::high_resolution_timer const timer;
        if (this_locality == 0)
        {
            // NOLINTNEXTLINE(bugprone-use-after-move)
            ft_data = scatter_to(scatter_direct_basename, std::move(send_data),
                num_sites_arg(num_localities), this_site_arg(this_locality),
                generation_arg(i + 1));
        }
        else
        {
            ft_data = scatter_from<std::vector<int>>(scatter_direct_basename,
                this_site_arg(this_locality), generation_arg(i + 1));
        }
        recv_data = ft_data.get();
        // Synchronize
        barrier.wait();
        // Write runtime into vector
        result[i] = timer.elapsed();

        // Check for correctness
        for (std::size_t check : recv_data)
        {
            HPX_TEST_EQ(42 + i + this_locality, check);
        }
    }

    if (this_locality == 0)
    {
        write_to_file(operation, "single_use", -1, num_localities, lpn,
            test_size, iterations, result);
    }
}

void test_one_shot_use_reduce(int lpn, std::size_t iterations, int test_size,
    std::string const& operation)
{
    // Get parameters
    std::size_t const num_localities =
        hpx::get_num_localities(hpx::launch::sync);
    std::size_t const this_locality = hpx::get_locality_id();
    // Ensure at least two localities
    HPX_TEST_LTE(static_cast<std::size_t>(2), num_localities);
    // Barrier for synchronization
    char const* const barrier_test_name = "/test/barrier/single";
    hpx::distributed::barrier barrier(barrier_test_name);
    // Result vector
    std::vector<double> result(iterations, 0.0);
    // Data
    std::vector<int> send_data;
    std::vector<int> recv_data;
    hpx::future<std::vector<int>> ft_data;

    for (std::size_t i = 0; i != iterations; ++i)
    {
        send_data = std::vector<int>(test_size, static_cast<int>(i));

        // Time collective
        hpx::chrono::high_resolution_timer const timer;
        if (this_locality == 0)
        {
            // NOLINTNEXTLINE(bugprone-use-after-move)
            ft_data = reduce_here(reduce_direct_basename, std::move(send_data),
                vector_adder{}, num_sites_arg(num_localities),
                this_site_arg(this_locality), generation_arg(i + 1));
            recv_data = ft_data.get();
        }
        else
        {
            hpx::future<void> finished =
                // NOLINTNEXTLINE(bugprone-use-after-move)
                reduce_there(reduce_direct_basename, std::move(send_data),
                    this_site_arg(this_locality), generation_arg(i + 1));
            finished.get();
        }
        // Synchronize
        barrier.wait();
        // Write runtime into vector
        result[i] = timer.elapsed();

        // Check for correctness
        if (this_locality == 0)
        {
            HPX_TEST_EQ(
                i * num_localities, static_cast<std::size_t>(recv_data[0]));
        }
    }

    if (this_locality == 0)
    {
        write_to_file(operation, "single_use", -1, num_localities, lpn,
            test_size, iterations, result);
    }
}

void test_one_shot_use_broadcast(int lpn, std::size_t iterations, int test_size,
    std::string const& operation)
{
    // Get parameters
    std::size_t const num_localities =
        hpx::get_num_localities(hpx::launch::sync);
    std::size_t const this_locality = hpx::get_locality_id();
    // Ensure at least two localities
    HPX_TEST_LTE(static_cast<std::size_t>(2), num_localities);
    // Barrier for synchronization
    char const* const barrier_test_name = "/test/barrier/single";
    hpx::distributed::barrier barrier(barrier_test_name);
    // Result vector
    std::vector<double> result(iterations, 0.0);
    // Data
    std::vector<int> send_data;
    std::vector<int> recv_data;
    hpx::future<std::vector<int>> ft_data;

    for (std::size_t i = 0; i != iterations; ++i)
    {
        if (this_locality == 0)
        {
            send_data = std::vector<int>(test_size, static_cast<int>(i));
        }

        // Time collective
        hpx::chrono::high_resolution_timer const timer;
        if (this_locality == 0)
        {
            ft_data = broadcast_to(broadcast_direct_basename,
                // NOLINTNEXTLINE(bugprone-use-after-move)
                std::move(send_data), num_sites_arg(num_localities),
                this_site_arg(this_locality), generation_arg(i + 1));
        }
        else
        {
            ft_data =
                broadcast_from<std::vector<int>>(broadcast_direct_basename,
                    this_site_arg(this_locality), generation_arg(i + 1));
        }
        recv_data = ft_data.get();
        // Synchronize
        barrier.wait();
        // Write runtime into vector
        result[i] = timer.elapsed();

        // Check for correctness
        if (this_locality == 0)
        {
            HPX_TEST_EQ(i, static_cast<std::size_t>(recv_data[0]));
        }
    }

    if (this_locality == 0)
    {
        write_to_file(operation, "single_use", -1, num_localities, lpn,
            test_size, iterations, result);
    }
}

void test_one_shot_use_gather(int lpn, std::size_t iterations, int test_size,
    std::string const& operation)
{
    // Get parameters
    std::size_t const num_localities =
        hpx::get_num_localities(hpx::launch::sync);
    std::size_t const this_locality = hpx::get_locality_id();
    // Ensure at least two localities
    HPX_TEST_LTE(static_cast<std::size_t>(2), num_localities);
    // Barrier for synchronization
    char const* const barrier_test_name = "/test/barrier/single";
    hpx::distributed::barrier barrier(barrier_test_name);
    // Result vector
    std::vector<double> result(iterations, 0.0);
    // Data
    std::vector<int> send_data;
    std::vector<std::vector<int>> recv_data;
    hpx::future<std::vector<std::vector<int>>> ft_data;

    for (std::size_t i = 0; i != iterations; ++i)
    {
        send_data =
            std::vector<int>(test_size, static_cast<int>(i + this_locality));

        // Time collective
        hpx::chrono::high_resolution_timer const timer;
        if (this_locality == 0)
        {
            // NOLINTNEXTLINE(bugprone-use-after-move)
            ft_data = gather_here(gather_direct_basename, std::move(send_data),
                num_sites_arg(num_localities), this_site_arg(this_locality),
                generation_arg(i + 1));
            recv_data = ft_data.get();
        }
        else
        {
            hpx::future<void> finished =
                // NOLINTNEXTLINE(bugprone-use-after-move)
                gather_there(gather_direct_basename, std::move(send_data),
                    this_site_arg(this_locality), generation_arg(i + 1));
            finished.get();
        }
        // Synchronize
        barrier.wait();
        // Write runtime into vector
        result[i] = timer.elapsed();

        // Check for correctness
        if (this_locality == 0)
        {
            for (int j = 0; j < static_cast<int>(num_localities); ++j)
            {
                HPX_TEST_EQ(i + j, static_cast<std::size_t>(recv_data[j][0]));
            }
        }
    }

    if (this_locality == 0)
    {
        write_to_file(operation, "single_use", -1, num_localities, lpn,
            test_size, iterations, result);
    }
}

////////////////////////////////////////////////////////////////////////////////////////
// Multi-use shot collectives
void test_multiple_use_with_generation_scatter(int lpn, std::size_t iterations,
    int test_size, std::string const& operation)
{
    // Get parameters
    std::size_t const num_localities =
        hpx::get_num_localities(hpx::launch::sync);
    std::size_t const this_locality = hpx::get_locality_id();
    // Ensure at least two localities
    HPX_TEST_LTE(static_cast<std::size_t>(2), num_localities);
    // Create communicator
    auto const scatter_direct_client =
        create_communicator(scatter_direct_basename,
            num_sites_arg(num_localities), this_site_arg(this_locality));
    // Barrier for synchronization
    char const* const barrier_test_name = "/test/barrier/generation";
    hpx::distributed::barrier barrier(barrier_test_name);
    // Result vector
    std::vector<double> result(iterations, 0.0);
    // Data
    std::vector<std::vector<int>> send_data;
    std::vector<int> recv_data;
    hpx::future<std::vector<int>> ft_data;

    for (std::size_t i = 0; i != iterations; ++i)
    {
        if (this_locality == 0)
        {
            send_data = generate_matrix(static_cast<int>(num_localities),
                test_size, static_cast<int>(42 + i));
        }

        // Time collective
        hpx::chrono::high_resolution_timer const timer;
        if (this_locality == 0)
        {
            // NOLINTNEXTLINE(bugprone-use-after-move)
            ft_data = scatter_to(scatter_direct_client, std::move(send_data),
                generation_arg(i + 1));
        }
        else
        {
            ft_data = scatter_from<std::vector<int>>(
                scatter_direct_client, generation_arg(i + 1));
        }
        recv_data = ft_data.get();
        // Synchronize
        barrier.wait();
        // Write runtime into vector
        result[i] = timer.elapsed();

        // Check for correctness
        for (std::size_t check : recv_data)
        {
            HPX_TEST_EQ(42 + i + this_locality, check);
        }
    }

    if (this_locality == 0)
    {
        write_to_file(operation, "multi_use", -1, num_localities, lpn,
            test_size, iterations, result);
    }
}

void test_multiple_use_with_generation_reduce(int lpn, std::size_t iterations,
    int test_size, std::string const& operation)
{
    // Get parameters
    std::size_t const num_localities =
        hpx::get_num_localities(hpx::launch::sync);
    std::size_t const this_locality = hpx::get_locality_id();
    // Ensure at least two localities
    HPX_TEST_LTE(static_cast<std::size_t>(2), num_localities);
    // Create communicator
    auto const reduce_direct_client =
        create_communicator(reduce_direct_basename,
            num_sites_arg(num_localities), this_site_arg(this_locality));
    // Barrier for synchronization
    char const* const barrier_test_name = "/test/barrier/generation";
    hpx::distributed::barrier barrier(barrier_test_name);
    // Result vector
    std::vector<double> result(iterations, 0.0);
    // Data
    std::vector<int> send_data;
    std::vector<int> recv_data;
    hpx::future<std::vector<int>> ft_data;

    for (std::size_t i = 0; i != iterations; ++i)
    {
        send_data = std::vector<int>(test_size, static_cast<int>(i));

        // Time collective
        hpx::chrono::high_resolution_timer const timer;
        if (this_locality == 0)
        {
            // NOLINTNEXTLINE(bugprone-use-after-move)
            ft_data = reduce_here(reduce_direct_client, std::move(send_data),
                vector_adder{}, generation_arg(i + 1));
            recv_data = ft_data.get();
        }
        else
        {
            hpx::future<void> finished = reduce_there(reduce_direct_client,
                // NOLINTNEXTLINE(bugprone-use-after-move)
                std::move(send_data), generation_arg(i + 1));
            finished.get();
        }
        // Synchronize
        barrier.wait();
        // Write runtime into vector
        result[i] = timer.elapsed();

        // Check for correctness
        if (this_locality == 0)
        {
            HPX_TEST_EQ(
                i * num_localities, static_cast<std::size_t>(recv_data[0]));
        }
    }

    if (this_locality == 0)
    {
        write_to_file(operation, "multi_use", -1, num_localities, lpn,
            test_size, iterations, result);
    }
}

void test_multiple_use_with_generation_broadcast(int lpn,
    std::size_t iterations, int test_size, std::string const& operation)
{
    // Get parameters
    std::size_t const num_localities =
        hpx::get_num_localities(hpx::launch::sync);
    std::size_t const this_locality = hpx::get_locality_id();
    // Ensure at least two localities
    HPX_TEST_LTE(static_cast<std::size_t>(2), num_localities);
    // Create communicator
    auto const broadcast_direct_client =
        create_communicator(broadcast_direct_basename,
            num_sites_arg(num_localities), this_site_arg(this_locality));
    // Barrier for synchronization
    char const* const barrier_test_name = "/test/barrier/generation";
    hpx::distributed::barrier barrier(barrier_test_name);
    // Result vector
    std::vector<double> result(iterations, 0.0);
    // Data
    std::vector<int> send_data;
    std::vector<int> recv_data;
    hpx::future<std::vector<int>> ft_data;

    for (std::size_t i = 0; i != iterations; ++i)
    {
        if (this_locality == 0)
        {
            send_data = std::vector<int>(test_size, static_cast<int>(i));
        }

        // Time collective
        hpx::chrono::high_resolution_timer const timer;
        if (this_locality == 0)
        {
            ft_data = broadcast_to(broadcast_direct_client,
                // NOLINTNEXTLINE(bugprone-use-after-move)
                std::move(send_data), generation_arg(i + 1));
        }
        else
        {
            ft_data = broadcast_from<std::vector<int>>(
                broadcast_direct_client, generation_arg(i + 1));
        }
        recv_data = ft_data.get();
        // Synchronize
        barrier.wait();
        // Write runtime into vector
        result[i] = timer.elapsed();

        // Check for correctness
        if (this_locality == 0)
        {
            HPX_TEST_EQ(i, static_cast<std::size_t>(recv_data[0]));
        }
    }

    if (this_locality == 0)
    {
        write_to_file(operation, "multi_use", -1, num_localities, lpn,
            test_size, iterations, result);
    }
}

void test_multiple_use_with_generation_gather(
    int lpn, std::size_t iterations, int test_size, std::string operation)
{
    // Get parameters
    std::size_t const num_localities =
        hpx::get_num_localities(hpx::launch::sync);
    std::size_t const this_locality = hpx::get_locality_id();
    // Ensure at least two localities
    HPX_TEST_LTE(static_cast<std::size_t>(2), num_localities);
    // Create communicator
    auto const gather_direct_client =
        create_communicator(gather_direct_basename,
            num_sites_arg(num_localities), this_site_arg(this_locality));
    // Barrier for synchronization
    char const* const barrier_test_name = "/test/barrier/generation";
    hpx::distributed::barrier barrier(barrier_test_name);
    // Result vector
    std::vector<double> result(iterations, 0.0);
    // Data
    std::vector<int> send_data;
    std::vector<std::vector<int>> recv_data;
    hpx::future<std::vector<std::vector<int>>> ft_data;

    for (std::size_t i = 0; i != iterations; ++i)
    {
        send_data =
            std::vector<int>(test_size, static_cast<int>(i + this_locality));

        // Time collective
        hpx::chrono::high_resolution_timer const timer;
        if (this_locality == 0)
        {
            // NOLINTNEXTLINE(bugprone-use-after-move)
            ft_data = gather_here(gather_direct_client, std::move(send_data),
                generation_arg(i + 1));
            recv_data = ft_data.get();
        }
        else
        {
            hpx::future<void> finished = gather_there(gather_direct_client,
                // NOLINTNEXTLINE(bugprone-use-after-move)
                std::move(send_data), generation_arg(i + 1));
            finished.get();
        }
        // Synchronize
        barrier.wait();
        // Write runtime into vector
        result[i] = timer.elapsed();

        // Check for correctness
        if (this_locality == 0)
        {
            for (int j = 0; j < static_cast<int>(num_localities); ++j)
            {
                HPX_TEST_EQ(i + j, static_cast<std::size_t>(recv_data[j][0]));
            }
        }
    }

    if (this_locality == 0)
    {
        write_to_file(operation, "multi_use", -1, num_localities, lpn,
            test_size, iterations, result);
    }
}

int hpx_main(hpx::program_options::variables_map& vm)
{
    int const arity = vm["arity"].as<int>();
    int const lpn = vm["lpn"].as<int>();
    int const test_size = vm["test_size"].as<int>();
    std::string const operation = vm["operation"].as<std::string>();
    int const iterations = vm["iterations"].as<int>();

    if (hpx::get_num_localities(hpx::launch::sync) > 1)
    {
        if (operation == "scatter")
        {
            if (arity == -1)
            {
                test_one_shot_use_scatter(
                    lpn, iterations, test_size, operation);
                test_multiple_use_with_generation_scatter(
                    lpn, iterations, test_size, operation);
            }
            else
            {
                test_scatter_hierarchical(
                    arity, lpn, iterations, test_size, operation);
            }
        }
        else if (operation == "reduce")
        {
            if (arity == -1)
            {
                test_one_shot_use_reduce(lpn, iterations, test_size, operation);
                test_multiple_use_with_generation_reduce(
                    lpn, iterations, test_size, operation);
            }
            else
            {
                test_reduce_hierarchical(
                    arity, lpn, iterations, test_size, operation);
            }
        }
        else if (operation == "broadcast")
        {
            if (arity == -1)
            {
                test_one_shot_use_broadcast(
                    lpn, iterations, test_size, operation);
                test_multiple_use_with_generation_broadcast(
                    lpn, iterations, test_size, operation);
            }
            else
            {
                test_broadcast_hierarchical(
                    arity, lpn, iterations, test_size, operation);
            }
        }
        else if (operation == "gather")
        {
            if (arity == -1)
            {
                test_one_shot_use_gather(lpn, iterations, test_size, operation);
                test_multiple_use_with_generation_gather(
                    lpn, iterations, test_size, operation);
            }
            else
            {
                test_gather_hierarchical(
                    arity, lpn, iterations, test_size, operation);
            }
        }
    }

    return hpx::finalize();
}

int main(int argc, char* argv[])
{
    using namespace hpx::program_options;

    // clang-format off
    options_description desc_commandline;
    desc_commandline.add_options()
        ("arity", value<int>()->default_value(-1), "Arity of the operation")
        ("lpn", value<int>()->default_value(2),
            "Number of localities per Node")
        ("test_size", value<int>()->default_value(2),
            "Number of Integers One Locality receives")
        ("iterations", value<int>()->default_value(10),
            "Number of Iteration the collective is executed")
        ("operation", value<std::string>()->default_value("scatter"),
            "Collective Operation (scatter, reduce, broadcast, gather)");
    // clang-format on

    std::vector<std::string> const cfg = {"hpx.run_hpx_main!=1"};

    hpx::init_params init_args;
    init_args.desc_cmdline = desc_commandline;
    init_args.cfg = cfg;

    HPX_TEST_EQ(hpx::init(argc, argv, init_args), 0);
    return hpx::util::report_errors();
}

#endif
