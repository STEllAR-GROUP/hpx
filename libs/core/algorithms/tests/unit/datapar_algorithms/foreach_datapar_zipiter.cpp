//  Copyright (c) 2016 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/algorithm.hpp>
#include <hpx/datapar.hpp>
#include <hpx/init.hpp>
#include <hpx/modules/testing.hpp>

#include <cstddef>
#include <iostream>
#include <iterator>
#include <string>
#include <utility>
#include <vector>

#include <hpx/modules/program_options.hpp>

#include "../algorithms/test_utils.hpp"

///////////////////////////////////////////////////////////////////////////////
struct set_42
{
    template <typename Tuple>
    void operator()(Tuple&& t)
    {
        hpx::get<0>(t) = 42;
        hpx::get<1>(t) = 42;
    }
};

///////////////////////////////////////////////////////////////////////////////
template <typename ExPolicy, typename IteratorTag>
void for_each_zipiter_test(ExPolicy&& policy, IteratorTag)
{
    static_assert(hpx::is_execution_policy<ExPolicy>::value,
        "hpx::is_execution_policy<ExPolicy>::value");

    typedef std::vector<int>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<int> c(10007), d(10007);
    std::iota(std::begin(c), std::end(c), std::rand());
    std::iota(std::begin(d), std::end(d), std::rand());

    auto begin = hpx::util::zip_iterator(
        iterator(std::begin(c)), iterator(std::begin(d)));
    auto end =
        hpx::util::zip_iterator(iterator(std::end(c)), iterator(std::end(d)));

    // auto result =
    hpx::for_each(std::forward<ExPolicy>(policy), begin, end, set_42());

    // HPX_TEST_EQ(result, end);

    // verify values
    std::size_t count = 0;
    std::for_each(std::begin(c), std::end(c), [&count](int v) -> void {
        HPX_TEST_EQ(v, int(42));
        ++count;
    });
    HPX_TEST_EQ(count, c.size());
}

template <typename IteratorTag>
void for_each_zipiter_test()
{
    using namespace hpx::execution;

    for_each_zipiter_test(par_simd, IteratorTag());
    //     test_for_each_async(par_simd(task), IteratorTag());
}

void for_each_zipiter_test()
{
    for_each_zipiter_test<std::random_access_iterator_tag>();
}

///////////////////////////////////////////////////////////////////////////////
int hpx_main(hpx::program_options::variables_map& vm)
{
    unsigned int seed = (unsigned int) std::time(nullptr);
    if (vm.count("seed"))
        seed = vm["seed"].as<unsigned int>();

    std::cout << "using seed: " << seed << std::endl;
    std::srand(seed);

    for_each_zipiter_test();

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
