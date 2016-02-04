//  Copyright (c) 2014-2015 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>
#include <hpx/hpx.hpp>
#include <hpx/include/parallel_scan.hpp>
#include <hpx/util/lightweight_test.hpp>

#include <boost/iterator/counting_iterator.hpp>
#include <boost/range/functions.hpp>

#include "test_utils.hpp"

///////////////////////////////////////////////////////////////////////////////
template <typename ExPolicy, typename IteratorTag>
void test_exclusive_scan2(ExPolicy policy, IteratorTag)
{
    static_assert(
        hpx::parallel::is_execution_policy<ExPolicy>::value,
        "hpx::parallel::is_execution_policy<ExPolicy>::value");

    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<std::size_t> c(10007);
    std::vector<std::size_t> d(c.size());
    std::fill(boost::begin(c), boost::end(c), std::size_t(1));

    std::size_t const val(0);
    hpx::parallel::exclusive_scan(policy,
        iterator(boost::begin(c)), iterator(boost::end(c)), boost::begin(d),
        val);

    // verify values
    std::vector<std::size_t> e(c.size());
    hpx::parallel::v1::detail::sequential_exclusive_scan(
        boost::begin(c), boost::end(c), boost::begin(e), val,
        std::plus<std::size_t>());

    HPX_TEST(std::equal(boost::begin(d), boost::end(d), boost::begin(e)));
}

template <typename ExPolicy, typename IteratorTag>
void test_exclusive_scan2_async(ExPolicy p, IteratorTag)
{
    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<std::size_t> c(10007);
    std::vector<std::size_t> d(c.size());
    std::fill(boost::begin(c), boost::end(c), std::size_t(1));

    std::size_t const val(0);
    hpx::future<void> f =
        hpx::parallel::exclusive_scan(p,
            iterator(boost::begin(c)), iterator(boost::end(c)), boost::begin(d),
            val);
    f.wait();

    // verify values
    std::vector<std::size_t> e(c.size());
    hpx::parallel::v1::detail::sequential_exclusive_scan(
        boost::begin(c), boost::end(c), boost::begin(e), val,
        std::plus<std::size_t>());

    HPX_TEST(std::equal(boost::begin(d), boost::end(d), boost::begin(e)));
}

template <typename IteratorTag>
void test_exclusive_scan2()
{
    using namespace hpx::parallel;

    test_exclusive_scan2(seq, IteratorTag());
    test_exclusive_scan2(par, IteratorTag());
    test_exclusive_scan2(par_vec, IteratorTag());

    test_exclusive_scan2_async(seq(task), IteratorTag());
    test_exclusive_scan2_async(par(task), IteratorTag());

    test_exclusive_scan2(execution_policy(seq), IteratorTag());
    test_exclusive_scan2(execution_policy(par), IteratorTag());
    test_exclusive_scan2(execution_policy(par_vec), IteratorTag());

    test_exclusive_scan2(execution_policy(seq(task)), IteratorTag());
    test_exclusive_scan2(execution_policy(par(task)), IteratorTag());
}

void exclusive_scan_test2()
{
    test_exclusive_scan2<std::random_access_iterator_tag>();
    test_exclusive_scan2<std::forward_iterator_tag>();
    test_exclusive_scan2<std::input_iterator_tag>();
}

///////////////////////////////////////////////////////////////////////////////
int hpx_main(boost::program_options::variables_map& vm)
{
    unsigned int seed = (unsigned int)std::time(0);
    if (vm.count("seed"))
        seed = vm["seed"].as<unsigned int>();

    std::cout << "using seed: " << seed << std::endl;
    std::srand(seed);

    exclusive_scan_test2();

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
    cfg.push_back("hpx.os_threads=" +
        boost::lexical_cast<std::string>(hpx::threads::hardware_concurrency()));

    // Initialize and run HPX
    HPX_TEST_EQ_MSG(hpx::init(desc_commandline, argc, argv, cfg), 0,
        "HPX main exited with non-zero status");

    return hpx::util::report_errors();
}
