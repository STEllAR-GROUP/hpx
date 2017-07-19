//  Copyright (c) 2017 Taeguk Kwon
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>
#include <hpx/hpx.hpp>
#include <hpx/include/parallel_unique.hpp>
#include <hpx/util/lightweight_test.hpp>

#include <boost/random.hpp>

#include <algorithm>
#include <cstddef>
#include <iostream>
#include <iterator>
#include <numeric>
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

    boost::random::mt19937 gen;
    boost::random::uniform_int_distribution<> dist;
};

////////////////////////////////////////////////////////////////////////////
template <typename ExPolicy, typename DataType>
void test_unique_copy(ExPolicy policy, DataType)
{
    static_assert(
        hpx::parallel::execution::is_execution_policy<ExPolicy>::value,
        "hpx::parallel::execution::is_execution_policy<ExPolicy>::value");

    using hpx::util::get;

    std::size_t const size = 10007;
    std::vector<DataType> c(size), dest_res(size), dest_sol(size);
    std::generate(std::begin(c), std::end(c), random_fill(0, 6));

    auto result = hpx::parallel::unique_copy(policy,
        c, std::begin(dest_res));
    auto solution = std::unique_copy(std::begin(c), std::end(c),
        std::begin(dest_sol));

    HPX_TEST(get<0>(result) == std::end(c));

    bool equality = std::equal(
        std::begin(dest_res), get<1>(result),
        std::begin(dest_sol), solution);

    HPX_TEST(equality);
}

template <typename ExPolicy, typename DataType>
void test_unique_copy_async(ExPolicy policy, DataType)
{
    static_assert(
        hpx::parallel::execution::is_execution_policy<ExPolicy>::value,
        "hpx::parallel::execution::is_execution_policy<ExPolicy>::value");

    using hpx::util::get;

    std::size_t const size = 10007;
    std::vector<DataType> c(size), dest_res(size), dest_sol(size);
    std::generate(std::begin(c), std::end(c), random_fill(0, 6));

    auto f = hpx::parallel::unique_copy(policy,
        c, std::begin(dest_res));
    auto result = f.get();
    auto solution = std::unique_copy(std::begin(c), std::end(c),
        std::begin(dest_sol));

    HPX_TEST(get<0>(result) == std::end(c));

    bool equality = std::equal(
        std::begin(dest_res), get<1>(result),
        std::begin(dest_sol), solution);

    HPX_TEST(equality);
}

#if defined(HPX_HAVE_ALGORITHM_INPUT_ITERATOR_SUPPORT)
template <typename ExPolicy, typename DataType>
void test_unique_copy_outiter(ExPolicy policy, DataType)
{
    static_assert(
        hpx::parallel::execution::is_execution_policy<ExPolicy>::value,
        "hpx::parallel::execution::is_execution_policy<ExPolicy>::value");

    using hpx::util::get;

    std::size_t const size = 10007;
    std::vector<DataType> c(size), dest_res(0), dest_sol(0);
    std::generate(std::begin(c), std::end(c), random_fill(0, 6));

    auto result = hpx::parallel::unique_copy(policy,
        c, std::back_inserter(dest_res));
    auto solution = std::unique_copy(std::begin(c), std::end(c),
        std::back_inserter(dest_sol));

    HPX_TEST(get<0>(result) == std::end(c));

    bool equality = std::equal(
        std::begin(dest_res), std::end(dest_res),
        std::begin(dest_sol), std::end(dest_sol));

    HPX_TEST(equality);
}

template <typename ExPolicy, typename DataType>
void test_unique_copy_outiter_async(ExPolicy policy, DataType)
{
    static_assert(
        hpx::parallel::execution::is_execution_policy<ExPolicy>::value,
        "hpx::parallel::execution::is_execution_policy<ExPolicy>::value");

    using hpx::util::get;

    std::size_t const size = 10007;
    std::vector<DataType> c(size), dest_res(0), dest_sol(0);
    std::generate(std::begin(c), std::end(c), random_fill(0, 6));

    auto f = hpx::parallel::unique_copy(policy,
        c, std::back_inserter(dest_res));
    auto result = f.get();
    auto solution = std::unique_copy(std::begin(c), std::end(c),
        std::back_inserter(dest_sol));

    HPX_TEST(get<0>(result) == std::end(c));

    bool equality = std::equal(
        std::begin(dest_res), std::end(dest_res),
        std::begin(dest_sol), std::end(dest_sol));

    HPX_TEST(equality);
}
#endif

template <typename DataType>
void test_unique_copy()
{
    using namespace hpx::parallel;

    test_unique_copy(execution::seq, DataType());
    test_unique_copy(execution::par, DataType());
    test_unique_copy(execution::par_unseq, DataType());

    test_unique_copy_async(execution::seq(execution::task), DataType());
    test_unique_copy_async(execution::par(execution::task), DataType());

#if defined(HPX_HAVE_GENERIC_EXECUTION_POLICY)
    test_unique_copy(execution_policy(execution::seq), DataType());
    test_unique_copy(execution_policy(execution::par), DataType());
    test_unique_copy(execution_policy(execution::par_unseq), DataType());

    test_unique_copy(execution_policy(execution::seq(execution::task)),
        DataType());
    test_unique_copy(execution_policy(execution::par(execution::task)),
        DataType());
#endif

#if defined(HPX_HAVE_ALGORITHM_INPUT_ITERATOR_SUPPORT)
    test_unique_copy_outiter(execution::seq, DataType());
    test_unique_copy_outiter(execution::par, DataType());
    test_unique_copy_outiter(execution::par_unseq, DataType());

    test_unique_copy_outiter_async(execution::seq(execution::task), DataType());
    test_unique_copy_outiter_async(execution::par(execution::task), DataType());

#if defined(HPX_HAVE_GENERIC_EXECUTION_POLICY)
    test_unique_copy_outiter(execution_policy(execution::seq), DataType());
    test_unique_copy_outiter(execution_policy(execution::par), DataType());
    test_unique_copy_outiter(execution_policy(execution::par_unseq), DataType());

    test_unique_copy_outiter(execution_policy(execution::seq(execution::task)),
        DataType());
    test_unique_copy_outiter(execution_policy(execution::par(execution::task)),
        DataType());
#endif
#endif
}

void test_unique_copy()
{
    test_unique_copy<int>();
    test_unique_copy<user_defined_type>();
}

int hpx_main(boost::program_options::variables_map& vm)
{
    unsigned int seed = (unsigned int)std::time(nullptr);
    if (vm.count("seed"))
        seed = vm["seed"].as<unsigned int>();

    std::cout << "using seed: " << seed << std::endl;
    std::srand(seed);

    test_unique_copy();
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
