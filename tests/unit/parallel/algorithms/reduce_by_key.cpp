//  Copyright (c) 2015-2016 John Biddiscombe
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>
#include <hpx/hpx.hpp>
#include <hpx/parallel/algorithm.hpp>
#include <hpx/parallel/algorithms/generate.hpp>
#include <hpx/parallel/algorithms/sort_by_key.hpp>
#include <hpx/parallel/algorithms/prefix_scan.hpp>
#include <hpx/parallel/algorithms/reduce_by_key.hpp>

// --seed=1451424610

// use smaller array sizes for debug tests
#if defined(HPX_DEBUG)
#define HPX_SORT_TEST_SIZE          131072
#define HPX_SORT_TEST_SIZE_STRINGS  50000
#endif

#include "sort_tests.hpp"

//
#define DEBUG_OUTPUT
//
namespace debug {
    template<typename T>
    void output(const std::string &name, const std::vector<T> &v) {
#ifdef DEBUG_OUTPUT
        std::cout << name.c_str() << "\t : {" << v.size() << "} : ";
        std::copy(std::begin(v), std::end(v), std::ostream_iterator<T>(std::cout, ", "));
        std::cout << "\n";
#endif
    }

    template<typename Iter>
    void output(const std::string &name, Iter begin, Iter end) {
#ifdef DEBUG_OUTPUT
        std::cout << name.c_str() << "\t : {" << std::distance(begin,end) << "} : ";
        std::copy(begin, end,
                  std::ostream_iterator<typename std::iterator_traits<Iter>::value_type>(std::cout, ", "));
        std::cout << "\n";
#endif
    }
};

////////////////////////////////////////////////////////////////////////////////
// call reduce_by_key with no comparison operator
template <typename ExPolicy, typename T>
void test_reduce_by_key1(ExPolicy && policy, T)
{
    static_assert(
            hpx::parallel::is_execution_policy<ExPolicy>::value,
            "hpx::parallel::is_execution_policy<ExPolicy>::value");
    msg(typeid(ExPolicy).name(), typeid(T).name(), "default", sync, random);
    std::cout << "\n";

    T rnd_min = (std::numeric_limits<T>::min)();
    T rnd_max = (std::numeric_limits<T>::max)();
    // just the value 1 for testing
    rnd_min = 1;
    rnd_max = 2;
    // Fill vector with random values
    std::vector<T> values(HPX_SORT_TEST_SIZE);
    rnd_fill<T>(values, rnd_min, rnd_max, T(std::rand()));

    // Fill vector with keys
    std::vector<T> keys(HPX_SORT_TEST_SIZE, 0);
    rnd_fill<T>(keys, 0, HPX_SORT_TEST_SIZE >> 8, T(std::rand()));
    std::sort(keys.begin(), keys.end());
    T key_min = *std::min_element(keys.begin(), keys.end());
    T key_max = *std::max_element(keys.begin(), keys.end());

    // output
    //debug::output<T>("\nkeys", keys);
    //debug::output<T>("\nvalues", values);

    auto policy2 = hpx::parallel::par.with(hpx::parallel::static_chunk_size(2));

    boost::uint64_t t = hpx::util::high_resolution_clock::now();
    // reduce_by_key, blocking when seq, par, par_vec
    hpx::parallel::reduce_by_key(
            policy2,
            //std::forward<ExPolicy>(policy),
            keys.begin(), keys.end(),
            values.begin(),
            keys.begin(),
            values.begin());
    boost::uint64_t elapsed = hpx::util::high_resolution_clock::now() - t;

    // output
    //debug::output<T>("\nkeys", keys);

    std::vector<T> check_values(HPX_SORT_TEST_SIZE);
    auto itb = check_values.begin()+1;
    for (int i=key_min; i<=key_max; i++) {
        int j = std::count(keys.begin(), keys.end(), i);
        //std::cout << "Num of " << i << " is " << j << "\n";
        auto ite = itb + j;
        if (ite>check_values.end()) ite=check_values.end();
        std::iota(itb,ite,1);
        itb = ite;
    }


    //debug::output<T>("\nvalues", values);
    //debug::output<T>("\nchecks", check_values);

    bool is_equal = std::equal(values.begin(), values.end(), check_values.begin());
    if (is_equal) {
        std::cout << "Test Passed\n";
    }
    HPX_TEST(is_equal);
}

////////////////////////////////////////////////////////////////////////////////
void test_reduce_by_key1()
{
    using namespace hpx::parallel;

    // default comparison operator (std::less)
//    test_reduce_by_key1(seq,     int());
    test_reduce_by_key1(par,     int());
//    test_reduce_by_key1(par_vec, int());
/*
    // default comparison operator (std::less)
    test_reduce_by_key1(seq,     double());
    test_reduce_by_key1(par,     double());
    test_reduce_by_key1(par_vec, double());

    // user supplied comparison operator (std::less)
    test_reduce_by_key1_comp(seq,     int(), std::less<std::size_t>());
    test_reduce_by_key1_comp(par,     int(), std::less<std::size_t>());
    test_reduce_by_key1_comp(par_vec, int(), std::less<std::size_t>());

    // user supplied comparison operator (std::greater)
    test_reduce_by_key1_comp(seq,     double(), std::greater<double>());
    test_reduce_by_key1_comp(par,     double(), std::greater<double>());
    test_reduce_by_key1_comp(par_vec, double(), std::greater<double>());

    // Async execution, default comparison operator
    test_reduce_by_key1_async(seq(task), int());
    test_reduce_by_key1_async(par(task), char());
    test_reduce_by_key1_async(seq(task), double());
    test_reduce_by_key1_async(par(task), float());
    test_reduce_by_key1_async_str(seq(task));
    test_reduce_by_key1_async_str(par(task));

    // Async execution, user comparison operator
    test_reduce_by_key1_async(seq(task), int(),    std::less<unsigned int>());
    test_reduce_by_key1_async(par(task), char(),   std::less<char>());
    //
    test_reduce_by_key1_async(seq(task), double(), std::greater<double>());
    test_reduce_by_key1_async(par(task), float(),  std::greater<float>());
    //
    test_reduce_by_key1_async_str(seq(task), std::greater<std::string>());
    test_reduce_by_key1_async_str(par(task), std::greater<std::string>());

    test_reduce_by_key1(execution_policy(seq),       int());
    test_reduce_by_key1(execution_policy(par),       int());
    test_reduce_by_key1(execution_policy(par_vec),   int());
    test_reduce_by_key1(execution_policy(seq(task)), int());
    test_reduce_by_key1(execution_policy(par(task)), int());
    test_reduce_by_key1(execution_policy(seq(task)), std::string());
    test_reduce_by_key1(execution_policy(par(task)), std::string());
*/
}

////////////////////////////////////////////////////////////////////////////////
int hpx_main(boost::program_options::variables_map& vm)
{
    unsigned int seed = (unsigned int)std::time(0);
    if (vm.count("seed"))
        seed = vm["seed"].as<unsigned int>();

    std::cout << "using seed: " << seed << std::endl;
    std::srand(seed);

    test_reduce_by_key1();
//    test_reduce_by_key2();
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

    HPX_TEST_EQ_MSG(hpx::init(desc_commandline, argc, argv, cfg), 0,
        "HPX main exited with non-zero status");

    return hpx::util::report_errors();
}
