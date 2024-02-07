//  Copyright (c) 2014-2020 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/init.hpp>

#include <iostream>
#include <string>
#include <vector>

#include "inclusive_scan_tests.hpp"

///////////////////////////////////////////////////////////////////////////////
template <typename IteratorTag>
void test_inclusive_scan1()
{
    using namespace hpx::execution;

    test_inclusive_scan1(IteratorTag());
    test_inclusive_scan1(seq, IteratorTag());
    test_inclusive_scan1(par, IteratorTag());
    test_inclusive_scan1(par_unseq, IteratorTag());

    test_inclusive_scan1_async(seq(task), IteratorTag());
    test_inclusive_scan1_async(par(task), IteratorTag());
}

void inclusive_scan_test1()
{
    test_inclusive_scan1<std::random_access_iterator_tag>();
    test_inclusive_scan1<std::forward_iterator_tag>();
}

///////////////////////////////////////////////////////////////////////////////
template <typename IteratorTag>
void test_inclusive_scan2()
{
    using namespace hpx::execution;

    test_inclusive_scan2(IteratorTag());
    test_inclusive_scan2(seq, IteratorTag());
    test_inclusive_scan2(par, IteratorTag());
    test_inclusive_scan2(par_unseq, IteratorTag());

    test_inclusive_scan2_async(seq(task), IteratorTag());
    test_inclusive_scan2_async(par(task), IteratorTag());
}

void inclusive_scan_test2()
{
    test_inclusive_scan2<std::random_access_iterator_tag>();
    test_inclusive_scan2<std::forward_iterator_tag>();
}

///////////////////////////////////////////////////////////////////////////////
template <typename IteratorTag>
void test_inclusive_scan3()
{
    using namespace hpx::execution;

    test_inclusive_scan3(IteratorTag());
    test_inclusive_scan3(seq, IteratorTag());
    test_inclusive_scan3(par, IteratorTag());
    test_inclusive_scan3(par_unseq, IteratorTag());

    test_inclusive_scan3_async(seq(task), IteratorTag());
    test_inclusive_scan3_async(par(task), IteratorTag());
}

void inclusive_scan_test3()
{
    test_inclusive_scan3<std::random_access_iterator_tag>();
    test_inclusive_scan3<std::forward_iterator_tag>();
}

////////////////////////////////////////////////////////////////////////////////
void inclusive_scan_validate()
{
    std::vector<int> a, b;
    // test scan algorithms using separate array for output
    //  std::cout << " Validating dual arrays " <<std::endl;
    test_inclusive_scan_validate(hpx::execution::seq, a, b);
    test_inclusive_scan_validate(hpx::execution::par, a, b);
    // test scan algorithms using same array for input and output
    //  std::cout << " Validating in_place arrays " <<std::endl;
    test_inclusive_scan_validate(hpx::execution::seq, a, a);
    test_inclusive_scan_validate(hpx::execution::par, a, a);
}

///////////////////////////////////////////////////////////////////////////////
void inclusive_scan_benchmark()
{
    try
    {
#if defined(HPX_DEBUG)
        std::vector<double> c(1000000);
#else
        std::vector<double> c(100000000);
#endif
        std::vector<double> d(c.size());
        std::fill(std::begin(c), std::end(c), 1.0);

        double const val(0);
        auto op = [](double v1, double v2) { return v1 + v2; };

        hpx::chrono::high_resolution_timer t;
        hpx::inclusive_scan(hpx::execution::par, std::begin(c), std::end(c),
            std::begin(d), op, val);
        double elapsed = t.elapsed();

        // verify values
        std::vector<double> e(c.size());
        hpx::parallel::detail::sequential_inclusive_scan(
            std::begin(c), std::end(c), std::begin(e), val, op);

        bool ok = std::equal(std::begin(d), std::end(d), std::begin(e));
        HPX_TEST(ok);
        if (ok)
        {
            // CDash graph plotting
            hpx::util::print_cdash_timing("InclusiveScanTime", elapsed);
        }
    }
    catch (...)
    {
        HPX_TEST(false);
    }
}

///////////////////////////////////////////////////////////////////////////////
int hpx_main(hpx::program_options::variables_map& vm)
{
    unsigned int seed = (unsigned int) std::time(nullptr);
    if (vm.count("seed"))
        seed = vm["seed"].as<unsigned int>();

    std::cout << "using seed: " << seed << std::endl;
    std::srand(seed);

    inclusive_scan_test1();
    inclusive_scan_test2();
    inclusive_scan_test3();

    inclusive_scan_validate();
    inclusive_scan_benchmark();

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

    // By default, this test should run on all available cores
    std::vector<std::string> const cfg = {"hpx.os_threads=all"};

    // Initialize and run HPX
    hpx::local::init_params init_args;
    init_args.desc_cmdline = desc_commandline;
    init_args.cfg = cfg;

    HPX_TEST_EQ_MSG(hpx::local::init(hpx_main, argc, argv, init_args), 0,
        "HPX main exited with non-zero status");

    return hpx::util::report_errors();
}
