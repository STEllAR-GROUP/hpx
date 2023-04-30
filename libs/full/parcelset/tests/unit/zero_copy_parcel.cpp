//  Copyright (c) 2023 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>

#if !defined(HPX_COMPUTE_DEVICE_CODE)
#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/modules/testing.hpp>

///////////////////////////////////////////////////////////////////////////////
constexpr std::size_t num_elements = 10000;

std::vector<char> test(
    std::vector<char> const& data1, std::vector<char> const& data2)
{
    std::vector<char> data = data1;
    data.reserve(data1.size() + data2.size());
    for (auto c : data2)
    {
        data.push_back(c);
    }
    return data;
}

HPX_PLAIN_ACTION(test)

char fill_char()
{
    return static_cast<char>(std::rand() % 255);
}

void test_zero_copy_parcel(hpx::id_type const& id)
{
    std::vector<char> data1(num_elements);
    std::vector<char> data2(num_elements);

    std::generate(data1.begin(), data1.end(), fill_char);
    std::generate(data2.begin(), data2.end(), fill_char);

    auto f = hpx::async(test_action(), id, data1, data2);

    std::vector<char> expected = data1;
    expected.reserve(data1.size() + data2.size());
    for (auto c : data2)
    {
        expected.push_back(c);
    }

    HPX_TEST(f.get() == expected);
}

///////////////////////////////////////////////////////////////////////////////
int hpx_main(hpx::program_options::variables_map& vm)
{
    unsigned int seed = static_cast<unsigned int>(std::time(nullptr));
    if (vm.count("seed"))
        seed = vm["seed"].as<unsigned int>();

    std::cout << "using seed: " << seed << std::endl;
    std::srand(seed);

    for (hpx::id_type const& id : hpx::find_remote_localities())
    {
        test_zero_copy_parcel(id);
    }
    return hpx::finalize();
}

int main(int argc, char* argv[])
{
    // add command line option which controls the random number generator seed
    using namespace hpx::program_options;
    options_description desc_commandline(
        "Usage: " HPX_APPLICATION_STRING " [options]");

    // clang-format off
    desc_commandline.add_options()
        ("seed,s", value<unsigned int>(),
         "the random number generator seed to use for this run")
        ;
    // clang-format on

    // Initialize and run HPX
    hpx::init_params init_args;
    init_args.desc_cmdline = desc_commandline;

    HPX_TEST_EQ_MSG(hpx::init(argc, argv, init_args), 0,
        "HPX main exited with non-zero status");

    return hpx::util::report_errors();
}

#endif
