//  Copyright (c) 2022 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#include <hpx/chrono.hpp>
#include <hpx/datastructures/detail/small_vector.hpp>
#include <hpx/functional.hpp>
#include <hpx/init.hpp>

#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
template <typename Container>
std::uint64_t measure(std::size_t repeat, std::size_t size)
{
    std::uint64_t start = hpx::chrono::high_resolution_clock::now();
    for (std::size_t i = 0; i != repeat; ++i)
    {
        Container cont;
        for (std::size_t i = 0; i != size; ++i)
        {
            cont.push_back(typename Container::value_type{});
        }
    }
    return (hpx::chrono::high_resolution_clock::now() - start) / repeat;
}

template <typename T, std::size_t N>
void compare(std::size_t repeat, std::size_t size)
{
    std::uint64_t time = measure<hpx::detail::small_vector<T, N>>(repeat, size);

    std::cout << "-----Average-(hpx::small_vector<" << typeid(T).name() << ", "
              << N << ">)------ \n"
              << std::left << "Average execution time : " << std::right
              << std::setw(8) << time / 1e9 << "\n";
}

int hpx_main(hpx::program_options::variables_map& vm)
{
    // pull values from cmd
    std::size_t repeat = vm["test_count"].as<std::size_t>();
    std::size_t size = vm["vector_size"].as<std::size_t>();

    std::cout << std::left
              << "----------------Parameters---------------------\n"
              << std::left
              << "Vector size                       : " << std::right
              << std::setw(8) << size << "\n"
              << std::left
              << "Number of tests                   : " << std::right
              << std::setw(8) << repeat << "\n"
              << std::left
              << "Display time in                   : " << std::right
              << std::setw(8) << "Seconds\n"
              << std::flush;

    compare<int, 1>(repeat, size);
    compare<int, 2>(repeat, size);
    compare<int, 4>(repeat, size);
    compare<int, 8>(repeat, size);
    compare<int, 16>(repeat, size);

    compare<hpx::move_only_function<void()>, 1>(repeat, size);
    compare<hpx::move_only_function<void()>, 2>(repeat, size);
    compare<hpx::move_only_function<void()>, 4>(repeat, size);
    compare<hpx::move_only_function<void()>, 8>(repeat, size);
    compare<hpx::move_only_function<void()>, 16>(repeat, size);

    return hpx::local::finalize();
}

///////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
    //initialize program
    std::vector<std::string> const cfg = {"hpx.os_threads=1"};

    using namespace hpx::program_options;

    options_description cmdline("usage: " HPX_APPLICATION_STRING " [options]");

    // clang-format off
    cmdline.add_options()
        ("vector_size", value<std::size_t>()->default_value(10),
            "size of vector")
        ("test_count", value<std::size_t>()->default_value(100000),
            "number of tests to be averaged")
        ;
    // clang-format on

    hpx::local::init_params init_args;
    init_args.desc_cmdline = cmdline;
    init_args.cfg = cfg;

    return hpx::local::init(hpx_main, argc, argv, init_args);
}
