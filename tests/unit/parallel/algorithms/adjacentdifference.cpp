//  Copyright (c) 2015 Daniel Bourgeois
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>
#include <hpx/hpx.hpp>
#include <hpx/include/parallel_adjacent_difference.hpp>
#include <hpx/util/lightweight_test.hpp>
#include <boost/range/functions.hpp>

#include <cstddef>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#include "test_utils.hpp"

////////////////////////////////////////////////////////////////////////////
template <typename ExPolicy>
void test_adjacent_difference(ExPolicy policy)
{
    static_assert(
        hpx::parallel::execution::is_execution_policy<ExPolicy>::value,
        "hpx::parallel::execution::is_execution_policy<ExPolicy>::value");

    std::vector<std::size_t> c = test::random_iota(10007);
    std::vector<std::size_t> d(10007);
    std::vector<std::size_t> d_ans(10007);

    auto it = hpx::parallel::adjacent_difference(policy, boost::begin(c),
        boost::end(c), boost::begin(d));
    std::adjacent_difference(boost::begin(c),
        boost::end(c), boost::begin(d_ans));

    HPX_TEST(std::equal(boost::begin(d), boost::end(d),
        boost::begin(d_ans), [](std::size_t lhs, std::size_t rhs) -> bool
        {
            return lhs == rhs;
        }));

    HPX_TEST(boost::end(d) == it);
}

template <typename ExPolicy>
void test_adjacent_difference_async(ExPolicy p)
{
    static_assert(
        hpx::parallel::execution::is_execution_policy<ExPolicy>::value,
        "hpx::parallel::execution::is_execution_policy<ExPolicy>::value");

    std::vector<std::size_t> c = test::random_iota(10007);
    std::vector<std::size_t> d(10007);
    std::vector<std::size_t> d_ans(10007);

    auto f_it = hpx::parallel::adjacent_difference(p, boost::begin(c),
        boost::end(c), boost::begin(d));
    std::adjacent_difference(boost::begin(c),
        boost::end(c), boost::begin(d_ans));

    f_it.wait();
    HPX_TEST(std::equal(boost::begin(d), boost::end(d),
        boost::begin(d_ans), [](std::size_t lhs, std::size_t rhs) -> bool
        {
            return lhs == rhs;
        }));

    HPX_TEST(boost::end(d) == f_it.get());
}

void adjacent_difference_test()
{
    using namespace hpx::parallel;
    test_adjacent_difference(execution::seq);
    test_adjacent_difference(execution::par);
    test_adjacent_difference(execution::par_unseq);

    test_adjacent_difference_async(execution::seq(execution::task));
    test_adjacent_difference_async(execution::par(execution::task));

#if defined(HPX_HAVE_GENERIC_EXECUTION_POLICY)
    test_adjacent_difference(execution_policy(execution::seq));
    test_adjacent_difference(execution_policy(execution::par));
    test_adjacent_difference(execution_policy(execution::par_unseq));

    test_adjacent_difference(execution_policy(execution::seq(execution::task)));
    test_adjacent_difference(execution_policy(execution::par(execution::task)));
#endif
}

int hpx_main(boost::program_options::variables_map& vm)
{
    unsigned int seed = (unsigned int)std::time(nullptr);
    if (vm.count("seed"))
        seed = vm["seed"].as<unsigned int>();

    std::cout << "using seed: " << seed << std::endl;
    std::srand(seed);

    adjacent_difference_test();
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
