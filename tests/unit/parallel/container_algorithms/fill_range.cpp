//  copyright (c) 2018 Christopher Ogle
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/include/parallel_fill.hpp>
#include <hpx/util/lightweight_test.hpp>

#include <cstddef>
#include <iostream>
#include <iterator>
#include <numeric>
#include <string>
#include <vector>

#include "test_utils.hpp"

////////////////////////////////////////////////////////////////////////////
template <typename ExPolicy, typename IteratorTag>
void test_fill(ExPolicy policy, IteratorTag)
{
    static_assert(
        hpx::parallel::execution::is_execution_policy<ExPolicy>::value,
        "hpx::parallel::execution::is_execution_policy<ExPolicy>::value");

    std::vector<std::size_t> c(10007);
    std::iota(std::begin(c), std::end(c), std::rand());

    hpx::parallel::fill(policy, c, 10);

    // verify values
    std::size_t count = 0;
    std::for_each(std::begin(c), std::end(c),
        [&count](std::size_t v) -> void {
            HPX_TEST_EQ(v, std::size_t(10));
            ++count;
    });

    HPX_TEST_EQ(count, c.size());
}

template <typename ExPolicy, typename IteratorTag>
void test_fill_async(ExPolicy p, IteratorTag)
{
    std::vector<std::size_t> c(10007);
    std::iota(std::begin(c), std::end(c), std::rand());

    hpx::future<void> f = hpx::parallel::fill(p, c, 10);
    f.wait();

    std::size_t count = 0;
    std::for_each(std::begin(c), std::end(c),
        [&count](std::size_t v) -> void {
            HPX_TEST_EQ(v, std::size_t(10));
            ++count;
    });

    HPX_TEST_EQ(count, c.size());
}

template <typename IteratorTag>
void test_fill()
{
    using namespace hpx::parallel;
    test_fill(execution::seq, IteratorTag());
    test_fill(execution::par, IteratorTag());
    test_fill(execution::par_unseq, IteratorTag());

    test_fill_async(execution::seq(execution::task), IteratorTag());
    test_fill_async(execution::par(execution::task), IteratorTag());
}

void fill_test()
{
    test_fill<std::random_access_iterator_tag>();
    test_fill<std::forward_iterator_tag>();
}

////////////////////////////////////////////////////////////////////////////
int hpx_main(boost::program_options::variables_map& vm)
{
    unsigned int seed = (unsigned int)std::time(nullptr);
    if (vm.count("seed"))
        seed = vm["seed"].as<unsigned int>();

    std::cout << "using seed: " << seed << std::endl;
    std::srand(seed);

    fill_test();
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
