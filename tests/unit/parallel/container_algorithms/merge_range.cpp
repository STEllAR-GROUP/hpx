//  Copyright (c) 2017 Taeguk Kwon
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>
#include <hpx/hpx.hpp>
#include <hpx/include/parallel_merge.hpp>
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
void test_merge(ExPolicy policy, DataType)
{
    static_assert(
        hpx::parallel::execution::is_execution_policy<ExPolicy>::value,
        "hpx::parallel::execution::is_execution_policy<ExPolicy>::value");

    using hpx::util::get;

    std::size_t const size1 = 300007, size2 = 123456;
    std::vector<DataType> src1(size1), src2(size2),
        dest_res(size1 + size2), dest_sol(size1 + size2);

    std::generate(std::begin(src1), std::end(src1), random_fill(0, 6));
    std::generate(std::begin(src2), std::end(src2), random_fill(0, 8));
    std::sort(std::begin(src1), std::end(src1));
    std::sort(std::begin(src2), std::end(src2));

    auto result = hpx::parallel::merge(policy,
        src1, src2, std::begin(dest_res));
    auto solution = std::merge(
        std::begin(src1), std::end(src1),
        std::begin(src2), std::end(src2),
        std::begin(dest_sol));

    HPX_TEST(get<0>(result) == std::end(src1));
    HPX_TEST(get<1>(result) == std::end(src2));

    bool equality = std::equal(
        std::begin(dest_res), get<2>(result),
        std::begin(dest_sol), solution);

    HPX_TEST(equality);
}

template <typename ExPolicy, typename DataType>
void test_merge_async(ExPolicy policy, DataType)
{
    static_assert(
        hpx::parallel::execution::is_execution_policy<ExPolicy>::value,
        "hpx::parallel::execution::is_execution_policy<ExPolicy>::value");

    using hpx::util::get;

    std::size_t const size1 = 300007, size2 = 123456;
    std::vector<DataType> src1(size1), src2(size2),
        dest_res(size1 + size2), dest_sol(size1 + size2);

    std::generate(std::begin(src1), std::end(src1), random_fill(0, 6));
    std::generate(std::begin(src2), std::end(src2), random_fill(0, 8));
    std::sort(std::begin(src1), std::end(src1));
    std::sort(std::begin(src2), std::end(src2));

    auto f = hpx::parallel::merge(policy,
        src1, src2, std::begin(dest_res));
    auto result = f.get();
    auto solution = std::merge(
        std::begin(src1), std::end(src1),
        std::begin(src2), std::end(src2),
        std::begin(dest_sol));

    HPX_TEST(get<0>(result) == std::end(src1));
    HPX_TEST(get<1>(result) == std::end(src2));

    bool equality = std::equal(
        std::begin(dest_res), get<2>(result),
        std::begin(dest_sol), solution);

    HPX_TEST(equality);
}

#if defined(HPX_HAVE_ALGORITHM_INPUT_ITERATOR_SUPPORT)
template <typename ExPolicy, typename DataType>
void test_merge_outiter(ExPolicy policy, DataType)
{
    static_assert(
        hpx::parallel::execution::is_execution_policy<ExPolicy>::value,
        "hpx::parallel::execution::is_execution_policy<ExPolicy>::value");

    using hpx::util::get;

    std::size_t const size1 = 300007, size2 = 123456;
    std::vector<DataType> src1(size1), src2(size2),
        dest_res(0), dest_sol(0);

    std::generate(std::begin(src1), std::end(src1), random_fill(0, 6));
    std::generate(std::begin(src2), std::end(src2), random_fill(0, 8));
    std::sort(std::begin(src1), std::end(src1));
    std::sort(std::begin(src2), std::end(src2));

    auto result = hpx::parallel::merge(policy,
        src1, src2, std::back_inserter(dest_res));
    auto solution = std::merge(
        std::begin(src1), std::end(src1),
        std::begin(src2), std::end(src2),
        std::back_inserter(dest_sol));

    HPX_TEST(get<0>(result) == std::end(src1));
    HPX_TEST(get<1>(result) == std::end(src2));

    bool equality = std::equal(
        std::begin(dest_res), std::end(dest_res),
        std::begin(dest_sol), std::end(dest_sol));

    HPX_TEST(equality);
}

template <typename ExPolicy, typename DataType>
void test_merge_outiter_async(ExPolicy policy, DataType)
{
    static_assert(
        hpx::parallel::execution::is_execution_policy<ExPolicy>::value,
        "hpx::parallel::execution::is_execution_policy<ExPolicy>::value");

    using hpx::util::get;

    std::size_t const size1 = 300007, size2 = 123456;
    std::vector<DataType> src1(size1), src2(size2),
        dest_res(0), dest_sol(0);

    std::generate(std::begin(src1), std::end(src1), random_fill(0, 6));
    std::generate(std::begin(src2), std::end(src2), random_fill(0, 8));
    std::sort(std::begin(src1), std::end(src1));
    std::sort(std::begin(src2), std::end(src2));

    auto f = hpx::parallel::merge(policy,
        src1, src2, std::back_inserter(dest_res));
    auto result = f.get();
    auto solution = std::merge(
        std::begin(src1), std::end(src1),
        std::begin(src2), std::end(src2),
        std::back_inserter(dest_sol));

    HPX_TEST(get<0>(result) == std::end(src1));
    HPX_TEST(get<1>(result) == std::end(src2));

    bool equality = std::equal(
        std::begin(dest_res), std::end(dest_res),
        std::begin(dest_sol), std::end(dest_sol));

    HPX_TEST(equality);
}
#endif

template <typename DataType>
void test_merge()
{
    using namespace hpx::parallel;

    test_merge(execution::seq, DataType());
    test_merge(execution::par, DataType());
    test_merge(execution::par_unseq, DataType());

    test_merge_async(execution::seq(execution::task), DataType());
    test_merge_async(execution::par(execution::task), DataType());

#if defined(HPX_HAVE_GENERIC_EXECUTION_POLICY)
    test_merge(execution_policy(execution::seq), DataType());
    test_merge(execution_policy(execution::par), DataType());
    test_merge(execution_policy(execution::par_unseq), DataType());

    test_merge(execution_policy(execution::seq(execution::task)),
        DataType());
    test_merge(execution_policy(execution::par(execution::task)),
        DataType());
#endif

#if defined(HPX_HAVE_ALGORITHM_INPUT_ITERATOR_SUPPORT)
    test_merge_outiter(execution::seq, DataType());
    test_merge_outiter(execution::par, DataType());
    test_merge_outiter(execution::par_unseq, DataType());

    test_merge_outiter_async(execution::seq(execution::task), DataType());
    test_merge_outiter_async(execution::par(execution::task), DataType());

#if defined(HPX_HAVE_GENERIC_EXECUTION_POLICY)
    test_merge_outiter(execution_policy(execution::seq), DataType());
    test_merge_outiter(execution_policy(execution::par), DataType());
    test_merge_outiter(execution_policy(execution::par_unseq), DataType());

    test_merge_outiter(execution_policy(execution::seq(execution::task)),
        DataType());
    test_merge_outiter(execution_policy(execution::par(execution::task)),
        DataType());
#endif
#endif
}

void test_merge()
{
    test_merge<int>();
    //test_merge<user_defined_type>();
}

int hpx_main(boost::program_options::variables_map& vm)
{
    unsigned int seed = (unsigned int)std::time(nullptr);
    if (vm.count("seed"))
        seed = vm["seed"].as<unsigned int>();

    std::cout << "using seed: " << seed << std::endl;
    std::srand(seed);

    test_merge();
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
