//  Copyright (c) 2017 Taeguk Kwon
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>
#include <hpx/hpx.hpp>
#include <hpx/include/parallel_is_heap.hpp>
#include <hpx/util/lightweight_test.hpp>

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
    user_defined_type(int rand_no) : val(rand_no) {}

    bool operator<(user_defined_type const& t) const
    {
        if (this->name < t.name)
            return true;
        else if (this->name > t.name)
            return false;
        else
            return this->val < t.val;
    }

    const user_defined_type& operator++()
    {
        static const std::vector<std::string> name_list = {
            "ABB", "ABC", "ACB", "BCA", "CAA", "CAAA", "CAAB"
        };
        name = name_list[std::rand() % name_list.size()];
        ++val;
        return *this;
    }

    std::string name;
    int val;
};

////////////////////////////////////////////////////////////////////////////
template <typename ExPolicy, typename DataType>
void test_is_heap_until(ExPolicy policy, DataType)
{
    static_assert(
        hpx::parallel::execution::is_execution_policy<ExPolicy>::value,
        "hpx::parallel::execution::is_execution_policy<ExPolicy>::value");

    using hpx::util::get;

    std::size_t const size = 10007;
    std::vector<DataType> c(size);
    std::iota(std::begin(c), std::end(c), DataType(std::rand()));

    auto heap_end_iter = std::next(std::begin(c), std::rand() % c.size());
    std::make_heap(std::begin(c), heap_end_iter);

    auto result = hpx::parallel::is_heap_until(policy, c);
    auto solution = std::is_heap_until(std::begin(c), std::end(c));

    HPX_TEST(result == solution);
}

template <typename ExPolicy, typename DataType>
void test_is_heap_until_async(ExPolicy policy, DataType)
{
    static_assert(
        hpx::parallel::execution::is_execution_policy<ExPolicy>::value,
        "hpx::parallel::execution::is_execution_policy<ExPolicy>::value");

    using hpx::util::get;

    std::size_t const size = 10007;
    std::vector<DataType> c(size);
    std::iota(std::begin(c), std::end(c), DataType(std::rand()));

    auto heap_end_iter = std::next(std::begin(c), std::rand() % c.size());
    std::make_heap(std::begin(c), heap_end_iter);

    auto f = hpx::parallel::is_heap_until(policy, c);
    auto result = f.get();
    auto solution = std::is_heap_until(std::begin(c), std::end(c));

    HPX_TEST(result == solution);
}

template <typename DataType>
void test_is_heap_until()
{
    using namespace hpx::parallel;

    test_is_heap_until(execution::seq, DataType());
    test_is_heap_until(execution::par, DataType());
    test_is_heap_until(execution::par_unseq, DataType());

    test_is_heap_until_async(execution::seq(execution::task), DataType());
    test_is_heap_until_async(execution::par(execution::task), DataType());

#if defined(HPX_HAVE_GENERIC_EXECUTION_POLICY)
    test_is_heap_until(execution_policy(execution::seq), DataType());
    test_is_heap_until(execution_policy(execution::par), DataType());
    test_is_heap_until(execution_policy(execution::par_unseq), DataType());

    test_is_heap_until(execution_policy(execution::seq(execution::task)),
        DataType());
    test_is_heap_until(execution_policy(execution::par(execution::task)),
        DataType());
#endif
}

void test_is_heap_until()
{
    test_is_heap_until<int>();
    test_is_heap_until<user_defined_type>();
}

int hpx_main(boost::program_options::variables_map& vm)
{
    unsigned int seed = (unsigned int)std::time(nullptr);
    if (vm.count("seed"))
        seed = vm["seed"].as<unsigned int>();

    std::cout << "using seed: " << seed << std::endl;
    std::srand(seed);

    test_is_heap_until();
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
