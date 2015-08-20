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
void test_exclusive_scan1(ExPolicy policy, IteratorTag)
{
    BOOST_STATIC_ASSERT(hpx::parallel::is_execution_policy<ExPolicy>::value);

    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<std::size_t> c(10007);
    std::vector<std::size_t> d(c.size());
    std::fill(boost::begin(c), boost::end(c), std::size_t(1));

    std::size_t const val(0);
    auto op =
        [val](std::size_t v1, std::size_t v2) {
            return v1 + v2;
        };

    hpx::parallel::exclusive_scan(policy,
        iterator(boost::begin(c)), iterator(boost::end(c)), boost::begin(d),
        val, op);

    // verify values
    std::vector<std::size_t> e(c.size());
    hpx::parallel::v1::detail::sequential_exclusive_scan(
        boost::begin(c), boost::end(c), boost::begin(e), val, op);

    HPX_TEST(std::equal(boost::begin(d), boost::end(d), boost::begin(e)));
}

template <typename ExPolicy, typename IteratorTag>
void test_exclusive_scan1_async(ExPolicy p, IteratorTag)
{
    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<std::size_t> c(10007);
    std::vector<std::size_t> d(c.size());
    std::fill(boost::begin(c), boost::end(c), std::size_t(1));

    std::size_t const val(0);
    auto op =
        [val](std::size_t v1, std::size_t v2) {
            return v1 + v2;
        };

    hpx::future<void> f =
        hpx::parallel::exclusive_scan(p,
            iterator(boost::begin(c)), iterator(boost::end(c)), boost::begin(d),
            val, op);
    f.wait();

    // verify values
    std::vector<std::size_t> e(c.size());
    hpx::parallel::v1::detail::sequential_exclusive_scan(
        boost::begin(c), boost::end(c), boost::begin(e), val, op);

    HPX_TEST(std::equal(boost::begin(d), boost::end(d), boost::begin(e)));
}

template <typename IteratorTag>
void test_exclusive_scan1()
{
    using namespace hpx::parallel;

    test_exclusive_scan1(seq, IteratorTag());
    test_exclusive_scan1(par, IteratorTag());
    test_exclusive_scan1(par_vec, IteratorTag());

    test_exclusive_scan1_async(seq(task), IteratorTag());
    test_exclusive_scan1_async(par(task), IteratorTag());

    test_exclusive_scan1(execution_policy(seq), IteratorTag());
    test_exclusive_scan1(execution_policy(par), IteratorTag());
    test_exclusive_scan1(execution_policy(par_vec), IteratorTag());

    test_exclusive_scan1(execution_policy(seq(task)), IteratorTag());
    test_exclusive_scan1(execution_policy(par(task)), IteratorTag());
}

void exclusive_scan_test1()
{
    test_exclusive_scan1<std::random_access_iterator_tag>();
    test_exclusive_scan1<std::forward_iterator_tag>();
    test_exclusive_scan1<std::input_iterator_tag>();
}

///////////////////////////////////////////////////////////////////////////////
template <typename ExPolicy, typename IteratorTag>
void test_exclusive_scan2(ExPolicy policy, IteratorTag)
{
    BOOST_STATIC_ASSERT(hpx::parallel::is_execution_policy<ExPolicy>::value);

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
template <typename ExPolicy, typename IteratorTag>
void test_exclusive_scan_exception(ExPolicy policy, IteratorTag)
{
    BOOST_STATIC_ASSERT(hpx::parallel::is_execution_policy<ExPolicy>::value);

    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<std::size_t> c(10007);
    std::vector<std::size_t> d(c.size());
    std::fill(boost::begin(c), boost::end(c), std::size_t(1));

    bool caught_exception = false;
    try {
        hpx::parallel::exclusive_scan(policy,
            iterator(boost::begin(c)), iterator(boost::end(c)),
            boost::begin(d), std::size_t(0),
            [](std::size_t v1, std::size_t v2)
            {
                return throw std::runtime_error("test"), v1 + v2;
            });

        HPX_TEST(false);
    }
    catch(hpx::exception_list const& e) {
        caught_exception = true;
        test::test_num_exceptions<ExPolicy, IteratorTag>::call(policy, e);
    }
    catch(...) {
        HPX_TEST(false);
    }

    HPX_TEST(caught_exception);
}

template <typename ExPolicy, typename IteratorTag>
void test_exclusive_scan_exception_async(ExPolicy p, IteratorTag)
{
    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<std::size_t> c(10007);
    std::vector<std::size_t> d(c.size());
    std::fill(boost::begin(c), boost::end(c), std::size_t(1));

    bool caught_exception = false;
    bool returned_from_algorithm = false;
    try {
        hpx::future<void> f =
            hpx::parallel::exclusive_scan(p,
                iterator(boost::begin(c)), iterator(boost::end(c)),
                boost::begin(d), std::size_t(0),
                [](std::size_t v1, std::size_t v2)
                {
                    return throw std::runtime_error("test"), v1 + v2;
                });

        returned_from_algorithm = true;
        f.get();

        HPX_TEST(false);
    }
    catch(hpx::exception_list const& e) {
        caught_exception = true;
        test::test_num_exceptions<ExPolicy, IteratorTag>::call(p, e);
    }
    catch(...) {
        HPX_TEST(false);
    }

    HPX_TEST(caught_exception);
    HPX_TEST(returned_from_algorithm);
}

template <typename IteratorTag>
void test_exclusive_scan_exception()
{
    using namespace hpx::parallel;

    // If the execution policy object is of type vector_execution_policy,
    // std::terminate shall be called. therefore we do not test exceptions
    // with a vector execution policy
    test_exclusive_scan_exception(seq, IteratorTag());
    test_exclusive_scan_exception(par, IteratorTag());

    test_exclusive_scan_exception_async(seq(task), IteratorTag());
    test_exclusive_scan_exception_async(par(task), IteratorTag());

    test_exclusive_scan_exception(execution_policy(seq), IteratorTag());
    test_exclusive_scan_exception(execution_policy(par), IteratorTag());

    test_exclusive_scan_exception(execution_policy(seq(task)), IteratorTag());
    test_exclusive_scan_exception(execution_policy(par(task)), IteratorTag());
}

void exclusive_scan_exception_test()
{
    test_exclusive_scan_exception<std::random_access_iterator_tag>();
    test_exclusive_scan_exception<std::forward_iterator_tag>();
    test_exclusive_scan_exception<std::input_iterator_tag>();
}

///////////////////////////////////////////////////////////////////////////////
template <typename ExPolicy, typename IteratorTag>
void test_exclusive_scan_bad_alloc(ExPolicy policy, IteratorTag)
{
    BOOST_STATIC_ASSERT(hpx::parallel::is_execution_policy<ExPolicy>::value);

    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<std::size_t> c(10007);
    std::vector<std::size_t> d(c.size());
    std::fill(boost::begin(c), boost::end(c), std::size_t(1));

    bool caught_exception = false;
    try {
        hpx::parallel::exclusive_scan(policy,
            iterator(boost::begin(c)), iterator(boost::end(c)),
            boost::begin(d), std::size_t(0),
            [](std::size_t v1, std::size_t v2)
            {
                return throw std::bad_alloc(), v1 + v2;
            });

        HPX_TEST(false);
    }
    catch(std::bad_alloc const&) {
        caught_exception = true;
    }
    catch(...) {
        HPX_TEST(false);
    }

    HPX_TEST(caught_exception);
}

template <typename ExPolicy, typename IteratorTag>
void test_exclusive_scan_bad_alloc_async(ExPolicy p, IteratorTag)
{
    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<std::size_t> c(10007);
    std::vector<std::size_t> d(c.size());
    std::fill(boost::begin(c), boost::end(c), std::size_t(1));

    bool caught_exception = false;
    bool returned_from_algorithm = false;
    try {
        hpx::future<void> f =
            hpx::parallel::exclusive_scan(p,
                iterator(boost::begin(c)), iterator(boost::end(c)),
                boost::begin(d), std::size_t(0),
                [](std::size_t v1, std::size_t v2)
                {
                    return throw std::bad_alloc(), v1 + v2;
                });

        returned_from_algorithm = true;
        f.get();

        HPX_TEST(false);
    }
    catch(std::bad_alloc const&) {
        caught_exception = true;
    }
    catch(...) {
        HPX_TEST(false);
    }

    HPX_TEST(caught_exception);
    HPX_TEST(returned_from_algorithm);
}

template <typename IteratorTag>
void test_exclusive_scan_bad_alloc()
{
    using namespace hpx::parallel;

    // If the execution policy object is of type vector_execution_policy,
    // std::terminate shall be called. therefore we do not test exceptions
    // with a vector execution policy
    test_exclusive_scan_bad_alloc(seq, IteratorTag());
    test_exclusive_scan_bad_alloc(par, IteratorTag());

    test_exclusive_scan_bad_alloc_async(seq(task), IteratorTag());
    test_exclusive_scan_bad_alloc_async(par(task), IteratorTag());

    test_exclusive_scan_bad_alloc(execution_policy(seq), IteratorTag());
    test_exclusive_scan_bad_alloc(execution_policy(par), IteratorTag());

    test_exclusive_scan_bad_alloc(execution_policy(seq(task)), IteratorTag());
    test_exclusive_scan_bad_alloc(execution_policy(par(task)), IteratorTag());
}

void exclusive_scan_bad_alloc_test()
{
    test_exclusive_scan_bad_alloc<std::random_access_iterator_tag>();
    test_exclusive_scan_bad_alloc<std::forward_iterator_tag>();
    test_exclusive_scan_bad_alloc<std::input_iterator_tag>();
}

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
        int value = b[i];
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
        int value = b[i];
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
        int value = b[i];
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

    exclusive_scan_test1();
    exclusive_scan_test2();

    exclusive_scan_exception_test();
    exclusive_scan_bad_alloc_test();

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
        boost::lexical_cast<std::string>(hpx::threads::hardware_concurrency()));

    // Initialize and run HPX
    HPX_TEST_EQ_MSG(hpx::init(desc_commandline, argc, argv, cfg), 0,
        "HPX main exited with non-zero status");

    return hpx::util::report_errors();
}
