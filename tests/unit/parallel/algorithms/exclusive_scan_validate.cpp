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

#include <string>
#include <vector>

#include "test_utils.hpp"

// uncomment to see some numbers from scan algorithm validation
// #define DUMP_VALUES
#define FILL_VALUE  10
#define ARRAY_SIZE  10000
#define INITIAL_VAL 50
#define DISPLAY     10 // for debug output
#ifdef DUMP_VALUES
  #define DEBUG_OUT(x) \
    std::cout << x << std::endl; \
  #endif
#else
  #define DEBUG_OUT(x)
#endif

// n'th value of sum of 1+2+3+...
int check_n_triangle(int n) {
    return n<0 ? 0 : (n)*(n+1)/2;
}

// n'th value of sum of x+x+x+...
int check_n_const(int n, int x) {
    return n<0 ? 0 : n*x;
}

// run scan algorithm, validate that output array hold expected answers.
template <typename ExPolicy>
void test_exclusive_scan_validate(ExPolicy p, std::vector<int> &a, std::vector<int> &b)
{
    using namespace hpx::parallel;
    typedef std::vector<int>::iterator Iter;

    // test 1, fill array with numbers counting from 0, then run scan algorithm
    a.clear();
    std::copy(boost::counting_iterator<int>(0),
        boost::counting_iterator<int>(ARRAY_SIZE), std::back_inserter(a));
#ifdef DUMP_VALUES
    std::cout << "\nValidating counting from 0 " << "\nInput : ";
    std::copy(a.begin(), a.begin()+DISPLAY, std::ostream_iterator<int>(std::cout, ", "));
    std::cout << " ... ";
    std::copy(a.end()-DISPLAY, a.end(), std::ostream_iterator<int>(std::cout, ", "));
#endif
    b.resize(a.size());
    hpx::parallel::exclusive_scan(p, a.begin(), a.end(), b.begin(), INITIAL_VAL,
                                  [](int bar, int baz){ return bar+baz; });
#ifdef DUMP_VALUES
    std::cout << "\nOutput : ";
    std::copy(b.begin(), b.begin()+DISPLAY, std::ostream_iterator<int>(std::cout, ", "));
    std::cout << " ... ";
    std::copy(b.end()-DISPLAY, b.end(), std::ostream_iterator<int>(std::cout, ", "));
#endif
    //
    for (int i=0; i<static_cast<int>(b.size()); ++i) {
        // counting from zero,
        int value = b[i]; //-V108
        int expected_value  = INITIAL_VAL + check_n_triangle(i-1);
        if (!HPX_TEST(value == expected_value)) break;
    }

    // test 2, fill array with numbers counting from 1, then run scan algorithm
    a.clear();
    std::copy(boost::counting_iterator<int>(1),
        boost::counting_iterator<int>(ARRAY_SIZE), std::back_inserter(a));
  #ifdef DUMP_VALUES
    std::cout << "\nValidating counting from 1 " << "\nInput : ";
    std::copy(a.begin(), a.begin()+DISPLAY, std::ostream_iterator<int>(std::cout, ", "));
    std::cout << " ... ";
    std::copy(a.end()-DISPLAY, a.end(), std::ostream_iterator<int>(std::cout, ", "));
  #endif
    b.resize(a.size());
    hpx::parallel::exclusive_scan(p, a.begin(), a.end(), b.begin(), INITIAL_VAL,
                                  [](int bar, int baz){ return bar+baz; });
  #ifdef DUMP_VALUES
    std::cout << "\nOutput : ";
    std::copy(b.begin(), b.begin()+DISPLAY, std::ostream_iterator<int>(std::cout, ", "));
    std::cout << " ... ";
    std::copy(b.end()-DISPLAY, b.end(), std::ostream_iterator<int>(std::cout, ", "));
  #endif
    //
    for (int i=0; i<static_cast<int>(b.size()); ++i) {
        // counting from 1, use i+1
        int value = b[i]; //-V108
        int expected_value  = INITIAL_VAL + check_n_triangle(i);
        if (!HPX_TEST(value == expected_value)) break;
    }

    // test 3, fill array with constant
    a.clear();
    std::fill_n(std::back_inserter(a), ARRAY_SIZE, FILL_VALUE);
  #ifdef DUMP_VALUES
    std::cout << "\nValidating constant values " << "\nInput : ";
    std::copy(a.begin(), a.begin()+DISPLAY, std::ostream_iterator<int>(std::cout, ", "));
    std::cout << " ... ";
    std::copy(a.end()-DISPLAY, a.end(), std::ostream_iterator<int>(std::cout, ", "));
  #endif
    b.resize(a.size());
    hpx::parallel::exclusive_scan(p, a.begin(), a.end(), b.begin(), INITIAL_VAL,
                                  [](int bar, int baz){ return bar+baz; });
  #ifdef DUMP_VALUES
    std::cout << "\nOutput : ";
    std::copy(b.begin(), b.begin()+DISPLAY, std::ostream_iterator<int>(std::cout, ", "));
    std::cout << " ... ";
    std::copy(b.end()-DISPLAY, b.end(), std::ostream_iterator<int>(std::cout, ", "));
    std::cout << std::endl;
  #endif
    //
    for (int i=0; i<static_cast<int>(b.size()); ++i) {
        // counting from zero,
        int value = b[i]; //-V108
        int expected_value  = INITIAL_VAL + check_n_const(i, FILL_VALUE);
        if (!HPX_TEST(value == expected_value)) break;
    }
}

void exclusive_scan_validate()
{
    std::vector<int> a, b;
    // test scan algorithms using separate array for output
    DEBUG_OUT("\nValidating separate arrays sequential");
    test_exclusive_scan_validate(hpx::parallel::seq, a, b);

    DEBUG_OUT("\nValidating separate arrays parallel");
    test_exclusive_scan_validate(hpx::parallel::par, a, b);

    // test scan algorithms using same array for input and output
    DEBUG_OUT("\nValidating in_place arrays sequential ");
    test_exclusive_scan_validate(hpx::parallel::seq, a, a);

    DEBUG_OUT("\nValidating in_place arrays parallel ");
    test_exclusive_scan_validate(hpx::parallel::par, a, a);
}

///////////////////////////////////////////////////////////////////////////////
int hpx_main(boost::program_options::variables_map& vm)
{
    unsigned int seed = (unsigned int)std::time(0);
    if (vm.count("seed"))
        seed = vm["seed"].as<unsigned int>();

    std::cout << "using seed: " << seed << std::endl;
    std::srand(seed);

    exclusive_scan_validate();

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
        std::to_string(hpx::threads::hardware_concurrency()));

    // Initialize and run HPX
    HPX_TEST_EQ_MSG(hpx::init(desc_commandline, argc, argv, cfg), 0,
        "HPX main exited with non-zero status");

    return hpx::util::report_errors();
}
