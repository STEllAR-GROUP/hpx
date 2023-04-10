//  Copyright (c) 2017-2018 Taeguk Kwon
//  Copyright (c) 2021 Akhil J Nair
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/algorithm.hpp>
#include <hpx/init.hpp>
#include <hpx/iterator_support/tests/iter_sent.hpp>
#include <hpx/modules/testing.hpp>

#include <algorithm>
#include <cstddef>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "test_utils.hpp"

////////////////////////////////////////////////////////////////////////////
struct user_defined_type
{
    user_defined_type() = default;
    user_defined_type(int rand_no)
      : val(rand_no)
      , name(name_list[std::rand() % name_list.size()])
    {
    }

    bool operator==(user_defined_type const& t) const
    {
        return this->name == t.name && this->val == t.val;
    }

    bool operator!=(user_defined_type const& t) const
    {
        return this->name != t.name || this->val != t.val;
    }

    static const std::vector<std::string> name_list;

    int val;
    std::string name;
};

const std::vector<std::string> user_defined_type::name_list{
    "ABB", "ABC", "ACB", "BASE", "CAA", "CAAA", "CAAB"};

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

////////////////////////////////////////////////////////////////////////////
void test_unique_copy_sent()
{
    using hpx::get;

    std::size_t const size = 10007;
    std::vector<std::size_t> c(size), dest_res(size), dest_sol(size);
    std::generate(std::begin(c), std::end(c), random_fill(0, 6));

    auto end_len = std::rand() % 10006 + 1;
    c[end_len] = 10;

    auto result = hpx::ranges::unique_copy(
        std::begin(c), sentinel<std::size_t>{10}, std::begin(dest_res));
    auto solution = std::unique_copy(
        std::begin(c), std::begin(c) + end_len, std::begin(dest_sol));

    HPX_TEST(result.in == std::next(std::begin(c), end_len));

    bool equality = test::equal(
        std::begin(dest_res), result.out, std::begin(dest_sol), solution);

    HPX_TEST(equality);
}

template <typename ExPolicy>
void test_unique_copy_sent(ExPolicy policy)
{
    using hpx::get;

    std::size_t const size = 10007;
    std::vector<std::size_t> c(size), dest_res(size), dest_sol(size);
    std::generate(std::begin(c), std::end(c), random_fill(0, 6));

    auto end_len = std::rand() % 10006 + 1;
    c[end_len] = 10;

    auto result = hpx::ranges::unique_copy(
        policy, std::begin(c), sentinel<std::size_t>{10}, std::begin(dest_res));
    auto solution = std::unique_copy(
        std::begin(c), std::begin(c) + end_len, std::begin(dest_sol));

    HPX_TEST(result.in == std::next(std::begin(c), end_len));

    bool equality = test::equal(
        std::begin(dest_res), result.out, std::begin(dest_sol), solution);

    HPX_TEST(equality);
}

////////////////////////////////////////////////////////////////////////////
template <typename DataType>
void test_unique_copy(DataType)
{
    using hpx::get;

    std::size_t const size = 10007;
    std::vector<DataType> c(size), dest_res(size), dest_sol(size);
    std::generate(std::begin(c), std::end(c), random_fill(0, 6));

    auto result = hpx::ranges::unique_copy(c, std::begin(dest_res));
    auto solution =
        std::unique_copy(std::begin(c), std::end(c), std::begin(dest_sol));

    HPX_TEST(result.in == std::end(c));

    bool equality = test::equal(
        std::begin(dest_res), result.out, std::begin(dest_sol), solution);

    HPX_TEST(equality);
}

template <typename ExPolicy, typename DataType>
void test_unique_copy(ExPolicy policy, DataType)
{
    static_assert(hpx::is_execution_policy<ExPolicy>::value,
        "hpx::is_execution_policy<ExPolicy>::value");

    using hpx::get;

    std::size_t const size = 10007;
    std::vector<DataType> c(size), dest_res(size), dest_sol(size);
    std::generate(std::begin(c), std::end(c), random_fill(0, 6));

    auto result = hpx::ranges::unique_copy(policy, c, std::begin(dest_res));
    auto solution =
        std::unique_copy(std::begin(c), std::end(c), std::begin(dest_sol));

    HPX_TEST(result.in == std::end(c));

    bool equality = test::equal(
        std::begin(dest_res), result.out, std::begin(dest_sol), solution);

    HPX_TEST(equality);
}

template <typename ExPolicy, typename DataType>
void test_unique_copy_async(ExPolicy policy, DataType)
{
    static_assert(hpx::is_execution_policy<ExPolicy>::value,
        "hpx::is_execution_policy<ExPolicy>::value");

    using hpx::get;

    std::size_t const size = 10007;
    std::vector<DataType> c(size), dest_res(size), dest_sol(size);
    std::generate(std::begin(c), std::end(c), random_fill(0, 6));

    auto f = hpx::ranges::unique_copy(policy, c, std::begin(dest_res));
    auto result = f.get();
    auto solution =
        std::unique_copy(std::begin(c), std::end(c), std::begin(dest_sol));

    HPX_TEST(result.in == std::end(c));

    bool equality = test::equal(
        std::begin(dest_res), result.out, std::begin(dest_sol), solution);

    HPX_TEST(equality);
}

template <typename DataType>
void test_unique_copy()
{
    test_unique_copy_sent();
    test_unique_copy_sent(hpx::execution::seq);
    test_unique_copy_sent(hpx::execution::par);
    test_unique_copy_sent(hpx::execution::par_unseq);

    test_unique_copy(DataType());
    test_unique_copy(hpx::execution::seq, DataType());
    test_unique_copy(hpx::execution::par, DataType());
    test_unique_copy(hpx::execution::par_unseq, DataType());

    test_unique_copy_async(
        hpx::execution::seq(hpx::execution::task), DataType());
    test_unique_copy_async(
        hpx::execution::par(hpx::execution::task), DataType());
}

void test_unique_copy()
{
    test_unique_copy<int>();
    test_unique_copy<user_defined_type>();
}

int hpx_main(hpx::program_options::variables_map& vm)
{
    unsigned int seed = (unsigned int) std::time(nullptr);
    if (vm.count("seed"))
        seed = vm["seed"].as<unsigned int>();

    std::cout << "using seed: " << seed << std::endl;
    std::srand(seed);

    test_unique_copy();
    return hpx::local::finalize();
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
    hpx::local::init_params init_args;
    init_args.desc_cmdline = desc_commandline;
    init_args.cfg = cfg;

    HPX_TEST_EQ_MSG(hpx::local::init(hpx_main, argc, argv, init_args), 0,
        "HPX main exited with non-zero status");

    return hpx::util::report_errors();
}
