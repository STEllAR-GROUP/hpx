//  Copyright (c) 2017 Taeguk Kwon
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>
#include <hpx/hpx.hpp>
#include <hpx/include/parallel_unique.hpp>
#include <hpx/util/lightweight_test.hpp>

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
      : val(rand_no),
        name(name_list[std::rand() % name_list.size()])
    {}

    bool operator<(int rand_base) const
    {
        static std::string const base_name = "BASE";

        if (this->name < base_name)
            return true;
        else if (this->name > base_name)
            return false;
        else
            return this->val < rand_base;
    }

    bool operator==(user_defined_type const& t) const
    {
        return this->name == t.name && this->val == t.val;
    }

    struct user_defined_type& operator++() { return *this; };

    static const std::vector<std::string> name_list;

    int val;
    std::string name;
};

const std::vector<std::string> user_defined_type::name_list{
    "ABB", "ABC", "ACB", "BASE", "CAA", "CAAA", "CAAB"
};

struct random_fill
{
    random_fill(int rand_base, int range)
      : gen(std::rand()),
        dist(rand_base - range / 2, rand_base + range / 2)
    {}

    int operator()()
    {
        return dist(gen);
    }

    std::mt19937 gen;
    std::uniform_int_distribution<> dist;
};

////////////////////////////////////////////////////////////////////////////
template <typename ExPolicy, typename DataType>
void test_unique(ExPolicy policy, DataType)
{
    static_assert(
        hpx::parallel::execution::is_execution_policy<ExPolicy>::value,
        "hpx::parallel::execution::is_execution_policy<ExPolicy>::value");

    using hpx::util::get;

    std::size_t const size = 10007;
    std::vector<DataType> c(size), d;
    std::generate(std::begin(c), std::end(c), random_fill(0, 6));
    d = c;

    auto result = hpx::parallel::unique(policy, c);
    auto solution = std::unique(std::begin(d), std::end(d));

    bool equality = std::equal(
        std::begin(c), result,
        std::begin(d), solution);

    HPX_TEST(equality);
}

template <typename ExPolicy, typename DataType>
void test_unique_async(ExPolicy policy, DataType)
{
    static_assert(
        hpx::parallel::execution::is_execution_policy<ExPolicy>::value,
        "hpx::parallel::execution::is_execution_policy<ExPolicy>::value");

    using hpx::util::get;

    std::size_t const size = 10007;
    std::vector<DataType> c(size), d;
    std::generate(std::begin(c), std::end(c), random_fill(0, 6));
    d = c;

    auto f = hpx::parallel::unique(policy, c);
    auto result = f.get();
    auto solution = std::unique(std::begin(d), std::end(d));

    bool equality = std::equal(
        std::begin(c), result,
        std::begin(d), solution);

    HPX_TEST(equality);
}

template <typename DataType>
void test_unique()
{
    using namespace hpx::parallel;

    test_unique(execution::seq, DataType());
    test_unique(execution::par, DataType());
    test_unique(execution::par_unseq, DataType());

    test_unique_async(execution::seq(execution::task), DataType());
    test_unique_async(execution::par(execution::task), DataType());

#if defined(HPX_HAVE_GENERIC_EXECUTION_POLICY)
    test_unique(execution_policy(execution::seq), DataType());
    test_unique(execution_policy(execution::par), DataType());
    test_unique(execution_policy(execution::par_unseq), DataType());

    test_unique(execution_policy(execution::seq(execution::task)),
        DataType());
    test_unique(execution_policy(execution::par(execution::task)),
        DataType());
#endif
}

void test_unique()
{
    test_unique<int>();
    test_unique<user_defined_type>();
}

int hpx_main(boost::program_options::variables_map& vm)
{
    unsigned int seed = (unsigned int)std::time(nullptr);
    if (vm.count("seed"))
        seed = vm["seed"].as<unsigned int>();

    std::cout << "using seed: " << seed << std::endl;
    std::srand(seed);

    test_unique();
    return hpx::finalize();
}

int main(int argc, char* argv[])
{
    // add command line option which controls the random number generator seed
    using namespace boost::program_options;
    options_description desc_commandline(
        "Usage: " HPX_APPLICATION_STRING " [options]");

    desc_commandline.add_options()
        ("seed,s", value<unsigned int>(),
        "the random number generator seed to use for this run")
        ;

    // By default this test should run on all available cores
    std::vector<std::string> const cfg = {
        "hpx.os_threads=all"
    };

    // Initialize and run HPX
    HPX_TEST_EQ_MSG(hpx::init(desc_commandline, argc, argv, cfg), 0,
        "HPX main exited with non-zero status");

    return hpx::util::report_errors();
}
