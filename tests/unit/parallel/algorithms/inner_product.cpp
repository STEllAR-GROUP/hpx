//  Copyright (c) 2015 Daniel Bourgeois
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>
#include <hpx/hpx.hpp>
#include <hpx/include/parallel_inner_product.hpp>
#include <hpx/util/lightweight_test.hpp>

#include <ctime>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#include "test_utils.hpp"

///////////////////////////////////////////////////////////////////////////////

template <typename ExPolicy, typename IteratorTag>
void test_inner_product(ExPolicy policy, IteratorTag)
{
    static_assert(
        hpx::parallel::is_execution_policy<ExPolicy>::value,
        "hpx::parallel::is_execution_policy<ExPolicy>::value");

    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<std::size_t> c = test::random_iota(1007);
    std::vector<std::size_t> d = test::random_iota(1007);
    std::size_t init = std::rand() % 1007; //-V101

    std::size_t r = hpx::parallel::inner_product(policy,
        iterator(boost::begin(c)), iterator(boost::end(c)),
        boost::begin(d), init);

    HPX_TEST_EQ(r, std::inner_product(
        boost::begin(c), boost::end(c), boost::begin(d), init));
}

template <typename ExPolicy, typename IteratorTag>
void test_inner_product_async(ExPolicy p, IteratorTag)
{
    static_assert(
        hpx::parallel::is_execution_policy<ExPolicy>::value,
        "hpx::parallel::is_execution_policy<ExPolicy>::value");

    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<std::size_t> c = test::random_iota(1007);
    std::vector<std::size_t> d = test::random_iota(1007);
    std::size_t init = std::rand() % 1007; //-V101

    hpx::future<std::size_t> fut_r =
        hpx::parallel::inner_product(p, iterator(boost::begin(c)),
        iterator(boost::end(c)), boost::begin(d), init);

    fut_r.wait();
    HPX_TEST_EQ(fut_r.get(), std::inner_product(
        boost::begin(c), boost::end(c), boost::begin(d), init));
}

template <typename IteratorTag>
void test_inner_product()
{
    using namespace hpx::parallel;

    test_inner_product(seq, IteratorTag());
    test_inner_product(par, IteratorTag());
    test_inner_product(par_vec, IteratorTag());

    test_inner_product_async(seq(task), IteratorTag());
    test_inner_product_async(par(task), IteratorTag());

#if defined(HPX_HAVE_GENERIC_EXECUTION_POLICY)
    test_inner_product(execution_policy(seq), IteratorTag());
    test_inner_product(execution_policy(par), IteratorTag());
    test_inner_product(execution_policy(par_vec), IteratorTag());

    test_inner_product(execution_policy(seq(task)), IteratorTag());
    test_inner_product(execution_policy(par(task)), IteratorTag());
#endif
}

void inner_product_test()
{
    test_inner_product<std::random_access_iterator_tag>();
    test_inner_product<std::forward_iterator_tag>();
}

///////////////////////////////////////////////////////////////////////////////
int hpx_main(boost::program_options::variables_map& vm)
{
    unsigned int seed = (unsigned int)std::time(nullptr);
    if (vm.count("seed"))
        seed = vm["seed"].as<unsigned int>();

    std::cout << "using seed: " << seed << std::endl;
    std::srand(seed);

    inner_product_test();

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
    std::vector<std::string> cfg;
    cfg.push_back("hpx.os_threads=all");

    // Initialize and run HPX
    HPX_TEST_EQ_MSG(hpx::init(desc_commandline, argc, argv, cfg), 0,
        "HPX main exited with non-zero status");

    return hpx::util::report_errors();
}
