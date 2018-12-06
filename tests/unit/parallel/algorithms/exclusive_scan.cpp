//  Copyright (c) 2014-2015 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>
#include <hpx/hpx.hpp>
#include <hpx/include/parallel_scan.hpp>
#include <hpx/util/lightweight_test.hpp>

#include <boost/iterator/counting_iterator.hpp>

#include <cstddef>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>

#include "test_utils.hpp"

///////////////////////////////////////////////////////////////////////////////
void exclusive_scan_benchmark()
{
    try {
      std::vector<double> c(100000000);
      std::vector<double> d(c.size());
      std::fill(std::begin(c), std::end(c), 1.0);

      double const val(0);
      auto op =
        [](double v1, double v2) {
          return v1 + v2;
      };

      hpx::util::high_resolution_timer t;
      hpx::parallel::exclusive_scan(hpx::parallel::execution::par,
        std::begin(c), std::end(c), std::begin(d),
        val, op);
      double elapsed = t.elapsed();

      // verify values
      std::vector<double> e(c.size());
      hpx::parallel::v1::detail::sequential_exclusive_scan(
          std::begin(c), std::end(c), std::begin(e), val, op);

      bool ok = std::equal(std::begin(d), std::end(d), std::begin(e));
      HPX_TEST(ok);
      if (ok) {
          // CDash graph plotting
          hpx::util::print_cdash_timing("ExclusiveScanTime", elapsed);
      }
    }
    catch (...) {
      HPX_TEST(false);
    }
}

///////////////////////////////////////////////////////////////////////////////
template <typename ExPolicy, typename IteratorTag>
void test_exclusive_scan1(ExPolicy policy, IteratorTag)
{
    static_assert(
        hpx::parallel::execution::is_execution_policy<ExPolicy>::value,
        "hpx::parallel::execution::is_execution_policy<ExPolicy>::value");

    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<std::size_t> c(10007);
    std::vector<std::size_t> d(c.size());
    std::fill(std::begin(c), std::end(c), std::size_t(1));

    std::size_t const val(0);
    auto op =
        [](std::size_t v1, std::size_t v2) {
            return v1 + v2;
        };

    hpx::parallel::exclusive_scan(policy,
        iterator(std::begin(c)), iterator(std::end(c)), std::begin(d),
        val, op);

    // verify values
    std::vector<std::size_t> e(c.size());
    hpx::parallel::v1::detail::sequential_exclusive_scan(
        std::begin(c), std::end(c), std::begin(e), val, op);

    HPX_TEST(std::equal(std::begin(d), std::end(d), std::begin(e)));
}

template <typename ExPolicy, typename IteratorTag>
void test_exclusive_scan1_async(ExPolicy p, IteratorTag)
{
    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<std::size_t> c(10007);
    std::vector<std::size_t> d(c.size());
    std::fill(std::begin(c), std::end(c), std::size_t(1));

    std::size_t const val(0);
    auto op =
        [](std::size_t v1, std::size_t v2) {
            return v1 + v2;
        };

    hpx::future<void> f =
        hpx::parallel::exclusive_scan(p,
            iterator(std::begin(c)), iterator(std::end(c)), std::begin(d),
            val, op);
    f.wait();

    // verify values
    std::vector<std::size_t> e(c.size());
    hpx::parallel::v1::detail::sequential_exclusive_scan(
        std::begin(c), std::end(c), std::begin(e), val, op);

    HPX_TEST(std::equal(std::begin(d), std::end(d), std::begin(e)));
}

template <typename IteratorTag>
void test_exclusive_scan1()
{
    using namespace hpx::parallel;

    test_exclusive_scan1(execution::seq, IteratorTag());
    test_exclusive_scan1(execution::par, IteratorTag());
    test_exclusive_scan1(execution::par_unseq, IteratorTag());

    test_exclusive_scan1_async(execution::seq(execution::task), IteratorTag());
    test_exclusive_scan1_async(execution::par(execution::task), IteratorTag());
}

void exclusive_scan_test1()
{
    test_exclusive_scan1<std::random_access_iterator_tag>();
    test_exclusive_scan1<std::forward_iterator_tag>();
#if defined(HPX_HAVE_ALGORITHM_INPUT_ITERATOR_SUPPORT)
    test_exclusive_scan1<std::input_iterator_tag>();
#endif
}

///////////////////////////////////////////////////////////////////////////////
int hpx_main(boost::program_options::variables_map& vm)
{
    unsigned int seed = (unsigned int)std::time(nullptr);
    if (vm.count("seed"))
        seed = vm["seed"].as<unsigned int>();

    std::cout << "using seed: " << seed << std::endl;
    std::srand(seed);

    exclusive_scan_test1();
    exclusive_scan_benchmark();

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
        "the random number generator seed to use for this run");

    // By default this test should run on all available cores
    std::vector<std::string> const cfg = {
        "hpx.os_threads=all"
    };

    // Initialize and run HPX
    HPX_TEST_EQ_MSG(hpx::init(desc_commandline, argc, argv, cfg), 0,
        "HPX main exited with non-zero status");

    return hpx::util::report_errors();
}
