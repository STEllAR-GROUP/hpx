//  Copyright (c) 2017-2018 Taeguk Kwon
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/include/parallel_merge.hpp>
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

    bool operator<(user_defined_type const& t) const
    {
        if (this->name < t.name)
            return true;
        else if (this->name > t.name)
            return false;
        else
            return this->val < t.val;
    }

    bool operator>(user_defined_type const& t) const
    {
        if (this->name > t.name)
            return true;
        else if (this->name < t.name)
            return false;
        else
            return this->val > t.val;
    }

    bool operator==(user_defined_type const& t) const
    {
        return this->name == t.name && this->val == t.val;
    }

    user_defined_type operator+(int val) const
    {
        user_defined_type t(*this);
        t.val += val;
        return t;
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
template <typename ExPolicy, typename DataType>
void test_inplace_merge(ExPolicy policy, DataType)
{
    static_assert(
        hpx::parallel::execution::is_execution_policy<ExPolicy>::value,
        "hpx::parallel::execution::is_execution_policy<ExPolicy>::value");

    std::size_t const left_size = 300007, right_size = 123456;
    std::vector<DataType> res(left_size + right_size), sol;

    auto res_first = std::begin(res);
    auto res_middle = res_first + left_size;
    auto res_last = std::end(res);

    std::generate(res_first, res_middle, random_fill(0, 6));
    std::generate(res_middle, res_last, random_fill(0, 8));
    std::sort(res_first, res_middle);
    std::sort(res_middle, res_last);

    sol = res;
    auto sol_first = std::begin(sol);
    auto sol_middle = sol_first + left_size;
    auto sol_last = std::end(sol);

    auto result = hpx::parallel::inplace_merge(policy, res, res_middle);
    std::inplace_merge(sol_first, sol_middle, sol_last);

    HPX_TEST(result == res_last);

    bool equality = test::equal(res_first, res_last, sol_first, sol_last);

    HPX_TEST(equality);
}

template <typename ExPolicy, typename DataType>
void test_inplace_merge_async(ExPolicy policy, DataType)
{
    static_assert(
        hpx::parallel::execution::is_execution_policy<ExPolicy>::value,
        "hpx::parallel::execution::is_execution_policy<ExPolicy>::value");

    std::size_t const left_size = 300007, right_size = 123456;
    std::vector<DataType> res(left_size + right_size), sol;

    auto res_first = std::begin(res);
    auto res_middle = res_first + left_size;
    auto res_last = std::end(res);

    std::generate(res_first, res_middle, random_fill(0, 6));
    std::generate(res_middle, res_last, random_fill(0, 8));
    std::sort(res_first, res_middle);
    std::sort(res_middle, res_last);

    sol = res;
    auto sol_first = std::begin(sol);
    auto sol_middle = sol_first + left_size;
    auto sol_last = std::end(sol);

    auto f = hpx::parallel::inplace_merge(policy, res, res_middle);
    auto result = f.get();
    std::inplace_merge(sol_first, sol_middle, sol_last);

    HPX_TEST(result == res_last);

    bool equality =
        test::equal(res_first, res_last, std::begin(sol), std::end(sol));

    HPX_TEST(equality);
}

///////////////////////////////////////////////////////////////////////////////
template <typename DataType>
void test_inplace_merge()
{
    using namespace hpx::parallel;

    test_inplace_merge(execution::seq, DataType());
    test_inplace_merge(execution::par, DataType());
    test_inplace_merge(execution::par_unseq, DataType());

    test_inplace_merge_async(execution::seq(execution::task), DataType());
    test_inplace_merge_async(execution::par(execution::task), DataType());
}

void test_inplace_merge()
{
    test_inplace_merge<int>();
    test_inplace_merge<user_defined_type>();
}

int hpx_main(hpx::program_options::variables_map& vm)
{
    unsigned int seed = (unsigned int) std::time(nullptr);
    if (vm.count("seed"))
        seed = vm["seed"].as<unsigned int>();

    std::cout << "using seed: " << seed << std::endl;
    std::srand(seed);

    test_inplace_merge();
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
