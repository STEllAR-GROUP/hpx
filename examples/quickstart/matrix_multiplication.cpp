////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2021 Dimitra Karatza
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

// Parallel implementation of matrix multiplication

#include <hpx/algorithm.hpp>
#include <hpx/execution.hpp>
#include <hpx/init.hpp>

#include <cstddef>
#include <iostream>
#include <random>
#include <vector>

std::mt19937 gen;

///////////////////////////////////////////////////////////////////////////////
//[mul_print_matrix
void print_matrix(std::vector<int> const& M, std::size_t rows, std::size_t cols,
    char const* message)
{
    std::cout << "\nMatrix " << message << " is:" << std::endl;
    for (std::size_t i = 0; i < rows; i++)
    {
        for (std::size_t j = 0; j < cols; j++)
            std::cout << M[i * cols + j] << " ";
        std::cout << "\n";
    }
}
//]

///////////////////////////////////////////////////////////////////////////////
//[mul_hpx_main
int hpx_main(hpx::program_options::variables_map& vm)
{
    using element_type = int;

    // Define matrix sizes
    std::size_t const rowsA = vm["n"].as<std::size_t>();
    std::size_t const colsA = vm["m"].as<std::size_t>();
    std::size_t const rowsB = colsA;
    std::size_t const colsB = vm["k"].as<std::size_t>();
    std::size_t const rowsR = rowsA;
    std::size_t const colsR = colsB;

    // Initialize matrices A and B
    std::vector<int> A(rowsA * colsA);
    std::vector<int> B(rowsB * colsB);
    std::vector<int> R(rowsR * colsR);

    // Define seed
    unsigned int seed = std::random_device{}();
    if (vm.count("seed"))
        seed = vm["seed"].as<unsigned int>();

    gen.seed(seed);
    std::cout << "using seed: " << seed << std::endl;

    // Define range of values
    int const lower = vm["l"].as<int>();
    int const upper = vm["u"].as<int>();

    // Matrices have random values in the range [lower, upper]
    std::uniform_int_distribution<element_type> dis(lower, upper);
    auto generator = std::bind(dis, gen);
    hpx::ranges::generate(A, generator);
    hpx::ranges::generate(B, generator);

    // Perform matrix multiplication
    hpx::experimental::for_loop(hpx::execution::par, 0, rowsA, [&](auto i) {
        hpx::experimental::for_loop(0, colsB, [&](auto j) {
            R[i * colsR + j] = 0;
            hpx::experimental::for_loop(0, rowsB, [&](auto k) {
                R[i * colsR + j] += A[i * colsA + k] * B[k * colsB + j];
            });
        });
    });

    // Print all 3 matrices
    print_matrix(A, rowsA, colsA, "A");
    print_matrix(B, rowsB, colsB, "B");
    print_matrix(R, rowsR, colsR, "R");

    return hpx::local::finalize();
}
//]

///////////////////////////////////////////////////////////////////////////////
//[mul_main
int main(int argc, char* argv[])
{
    using namespace hpx::program_options;
    options_description cmdline("usage: " HPX_APPLICATION_STRING " [options]");
    // clang-format off
    cmdline.add_options()
        ("n",
        hpx::program_options::value<std::size_t>()->default_value(2),
        "Number of rows of first matrix")
        ("m",
        hpx::program_options::value<std::size_t>()->default_value(3),
        "Number of columns of first matrix (equal to the number of rows of "
        "second matrix)")
        ("k",
        hpx::program_options::value<std::size_t>()->default_value(2),
        "Number of columns of second matrix")
        ("seed,s",
        hpx::program_options::value<unsigned int>(),
        "The random number generator seed to use for this run")
        ("l",
        hpx::program_options::value<int>()->default_value(0),
        "Lower limit of range of values")
        ("u",
        hpx::program_options::value<int>()->default_value(10),
        "Upper limit of range of values");
    // clang-format on
    hpx::local::init_params init_args;
    init_args.desc_cmdline = cmdline;

    return hpx::local::init(hpx_main, argc, argv, init_args);
}
//]
