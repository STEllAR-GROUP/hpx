//  Copyright (c) 2018 Christopher Ogle

//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/include/parallel_count.hpp>
#include <hpx/modules/testing.hpp>
#include <hpx/parallel/util/projection_identity.hpp>

#include <algorithm>
#include <cstddef>
#include <iostream>
#include <iterator>
#include <random>
#include <string>
#include <vector>

#include "test_utils.hpp"

struct user_defined_struct
{
    user_defined_struct() = default;
    user_defined_struct(int val)
      : val(val)
    {
    }

    ~user_defined_struct() = default;

    bool operator==(const user_defined_struct& rhs) const
    {
        return val == rhs.val;
    }

    int val;
};

struct random_fill
{
    random_fill(int rand_base, int range)
      : gen(std::rand())
      , dist(rand_base - range / 2, rand_base + range / 2)
    {
    }

    int operator()()
    {
        return dist(gen);
    }

    std::mt19937 gen;
    std::uniform_int_distribution<> dist;
};

template <typename IteratorTag, typename DataType>
void test_count(IteratorTag, DataType)
{
    std::vector<DataType> c{10007};
    std::generate(std::begin(c), std::end(c), random_fill(0, 20));

    const auto search_val = DataType(10);

    auto result = hpx::ranges::count(c, search_val);
    auto expected = std::count(std::begin(c), std::end(c), search_val);

    HPX_TEST_EQ(expected, result);
}

template <typename ExPolicy, typename IteratorTag, typename DataType>
void test_count(ExPolicy&& policy, IteratorTag, DataType)
{
    static_assert(
        hpx::parallel::execution::is_execution_policy<ExPolicy>::value,
        "hpx::parallel::execution::is_execution_policy<ExPolicy>::value");

    std::vector<DataType> c{10007};
    std::generate(std::begin(c), std::end(c), random_fill(0, 20));

    const auto search_val = DataType(10);

    auto result = hpx::ranges::count(policy, c, search_val);
    auto expected = std::count(std::begin(c), std::end(c), search_val);

    HPX_TEST_EQ(expected, result);
}

template <typename ExPolicy, typename IteratorTag, typename DataType>
void test_count_async(ExPolicy&& policy, IteratorTag, DataType)
{
    static_assert(
        hpx::parallel::execution::is_execution_policy<ExPolicy>::value,
        "hpx::parallel::execution::is_execution_policy<ExPolicy>::value");

    std::vector<DataType> c{10007};
    std::generate(std::begin(c), std::end(c), random_fill(0, 20));

    const auto search_val = DataType(10);

    auto f = hpx::ranges::count(policy, c, search_val);
    auto result = f.get();

    auto expected = std::count(std::begin(c), std::end(c), search_val);

    HPX_TEST_EQ(expected, result);
}

template <typename IteratorTag, typename DataType>
void test_count()
{
    using namespace hpx::parallel;

    test_count(IteratorTag(), DataType());

    auto seq = execution::seq;
    auto par = execution::par;
    auto par_unseq = execution::par_unseq;

    test_count(seq, IteratorTag(), DataType());
    test_count(par, IteratorTag(), DataType());
    test_count(par_unseq, IteratorTag(), DataType());

    test_count_async(
        execution::seq(execution::task), IteratorTag(), DataType());
    test_count_async(
        execution::par(execution::task), IteratorTag(), DataType());
}

void count_test()
{
    test_count<std::random_access_iterator_tag, int>();
    test_count<std::random_access_iterator_tag, user_defined_struct>();
}

///////////////////////////////////////////////////////////////////////////////
int hpx_main(hpx::program_options::variables_map& vm)
{
    unsigned int seed = (unsigned int) std::time(nullptr);
    if (vm.count("seed"))
        seed = vm["seed"].as<unsigned int>();

    std::cout << "using seed: " << seed << std::endl;
    std::srand(seed);

    count_test();
    return hpx::finalize();
}

int main(int argc, char* argv[])
{
    // add command line option which controls the random number generator seed
    using namespace hpx::program_options;
    options_description desc_commandline(
        "Usage: " HPX_APPLICATION_STRING " [options]");

    desc_commandline.add_options()("seed,s", value<unsigned int>(),
        "the random number generator seed to use for this run");

    // By default this test should run on all available cores
    std::vector<std::string> const cfg = {"hpx.os_threads=all"};

    // Initialize and run HPX
    HPX_TEST_EQ_MSG(hpx::init(desc_commandline, argc, argv, cfg), 0,
        "HPX main exited with non-zero status");

    return hpx::util::report_errors();
}
