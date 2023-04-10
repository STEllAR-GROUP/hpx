//  Copyright (c) 2022 Srinivas Yadav
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/algorithm.hpp>
#include <hpx/datapar.hpp>
#include <hpx/init.hpp>
#include <hpx/modules/testing.hpp>

#include <algorithm>
#include <cstddef>
#include <iostream>
#include <iterator>
#include <numeric>
#include <string>
#include <vector>

#include "../algorithms/test_utils.hpp"

///////////////////////////////////////////////////////////////////////////////
template <typename ExPolicy, typename IteratorTag>
void test_transform_reduce(ExPolicy&& policy, IteratorTag)
{
    static_assert(hpx::is_execution_policy<ExPolicy>::value,
        "hpx::is_execution_policy<ExPolicy>::value");

    typedef std::vector<int>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<int> c(10007);
    std::iota(std::begin(c), std::end(c), std::rand());

    auto reduce_op = [](auto v1, auto v2) { return v1 + v2; };

    auto convert_op = [](auto val) { return val * val; };

    int const init = int(1);

    int r1 = hpx::transform_reduce(policy, iterator(std::begin(c)),
        iterator(std::end(c)), init, reduce_op, convert_op);

    // verify values
    int r2 = std::accumulate(std::begin(c), std::end(c), init,
        [&reduce_op, &convert_op](
            auto res, auto val) { return reduce_op(res, convert_op(val)); });

    HPX_TEST_EQ(r1, r2);
    HPX_TEST_EQ(r1, r2);
}

template <typename ExPolicy, typename IteratorTag>
void test_transform_reduce_async(ExPolicy&& p, IteratorTag)
{
    typedef std::vector<int>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<int> c(10007);
    std::iota(std::begin(c), std::end(c), std::rand());

    int val = 42;
    auto op = std::plus<>{};

    hpx::future<int> f = hpx::transform_reduce(p, iterator(std::begin(c)),
        iterator(std::end(c)), val, op, [](auto v) { return v; });
    f.wait();

    // verify values
    int r2 = std::accumulate(std::begin(c), std::end(c), val, op);
    HPX_TEST_EQ(f.get(), r2);
}

template <typename IteratorTag>
void test_transform_reduce()
{
    using namespace hpx::execution;

    test_transform_reduce(simd, IteratorTag());
    test_transform_reduce(par_simd, IteratorTag());

    test_transform_reduce_async(simd(task), IteratorTag());
    test_transform_reduce_async(par_simd(task), IteratorTag());
}

void transform_reduce_test()
{
    test_transform_reduce<std::random_access_iterator_tag>();
    test_transform_reduce<std::forward_iterator_tag>();
}

int hpx_main(hpx::program_options::variables_map& vm)
{
    unsigned int seed = (unsigned int) std::time(nullptr);
    if (vm.count("seed"))
        seed = vm["seed"].as<unsigned int>();

    std::cout << "using seed: " << seed << std::endl;
    std::srand(seed);

    transform_reduce_test();
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

    //By default run on all available cores
    std::vector<std::string> const cfg = {"hpx.os_threads=all"};

    // Initialize and run HPX
    hpx::local::init_params init_args;
    init_args.desc_cmdline = desc_commandline;
    init_args.cfg = cfg;

    HPX_TEST_EQ_MSG(hpx::local::init(hpx_main, argc, argv, init_args), 0,
        "HPX main exited with non-zero status");

    return hpx::util::report_errors();
}
