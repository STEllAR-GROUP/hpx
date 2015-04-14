//  Copyright (c) 2014-2015 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>
#include <hpx/hpx.hpp>
#include <hpx/include/parallel_scan.hpp>
#include <hpx/util/lightweight_test.hpp>
//
#include <boost/iterator/counting_iterator.hpp>
#include <boost/range/functions.hpp>

#include "test_utils.hpp"

///////////////////////////////////////////////////////////////////////////////
template <typename ExPolicy, typename IteratorTag>
void test_inclusive_scan1(ExPolicy const& policy, IteratorTag)
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

    hpx::parallel::inclusive_scan(policy,
        iterator(boost::begin(c)), iterator(boost::end(c)), boost::begin(d),
        val, op);

    // verify values
    std::vector<std::size_t> e(c.size());
    hpx::parallel::detail::sequential_inclusive_scan(
        boost::begin(c), boost::end(c), boost::begin(e), val, op);

    HPX_TEST(std::equal(boost::begin(d), boost::end(d), boost::begin(e)));
}

template <typename ExPolicy, typename IteratorTag>
void test_inclusive_scan1_async(ExPolicy const& p, IteratorTag)
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
        hpx::parallel::inclusive_scan(p,
            iterator(boost::begin(c)), iterator(boost::end(c)), boost::begin(d),
            val, op);
    f.wait();

    // verify values
    std::vector<std::size_t> e(c.size());
    hpx::parallel::detail::sequential_inclusive_scan(
        boost::begin(c), boost::end(c), boost::begin(e), val, op);

    HPX_TEST(std::equal(boost::begin(d), boost::end(d), boost::begin(e)));
}

template <typename IteratorTag>
void test_inclusive_scan1()
{
    using namespace hpx::parallel;

    test_inclusive_scan1(seq, IteratorTag());
    test_inclusive_scan1(par, IteratorTag());
    test_inclusive_scan1(par_vec, IteratorTag());

    test_inclusive_scan1_async(seq(task), IteratorTag());
    test_inclusive_scan1_async(par(task), IteratorTag());

    test_inclusive_scan1(execution_policy(seq), IteratorTag());
    test_inclusive_scan1(execution_policy(par), IteratorTag());
    test_inclusive_scan1(execution_policy(par_vec), IteratorTag());

    test_inclusive_scan1(execution_policy(seq(task)), IteratorTag());
    test_inclusive_scan1(execution_policy(par(task)), IteratorTag());
}

void inclusive_scan_test1()
{
    test_inclusive_scan1<std::random_access_iterator_tag>();
    test_inclusive_scan1<std::forward_iterator_tag>();
    test_inclusive_scan1<std::input_iterator_tag>();
}

///////////////////////////////////////////////////////////////////////////////
template <typename ExPolicy, typename IteratorTag>
void test_inclusive_scan2(ExPolicy const& policy, IteratorTag)
{
    BOOST_STATIC_ASSERT(hpx::parallel::is_execution_policy<ExPolicy>::value);

    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<std::size_t> c(10007);
    std::vector<std::size_t> d(c.size());
    std::fill(boost::begin(c), boost::end(c), std::size_t(1));

    std::size_t const val(0);
    hpx::parallel::inclusive_scan(policy,
        iterator(boost::begin(c)), iterator(boost::end(c)), boost::begin(d),
        val);

    // verify values
    std::vector<std::size_t> e(c.size());
    hpx::parallel::detail::sequential_inclusive_scan(
        boost::begin(c), boost::end(c), boost::begin(e), val,
        std::plus<std::size_t>());

    HPX_TEST(std::equal(boost::begin(d), boost::end(d), boost::begin(e)));
}

template <typename ExPolicy, typename IteratorTag>
void test_inclusive_scan2_async(ExPolicy const& p, IteratorTag)
{
    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<std::size_t> c(10007);
    std::vector<std::size_t> d(c.size());
    std::fill(boost::begin(c), boost::end(c), std::size_t(1));

    std::size_t const val(0);
    hpx::future<void> f =
        hpx::parallel::inclusive_scan(p,
            iterator(boost::begin(c)), iterator(boost::end(c)), boost::begin(d),
            val);
    f.wait();

    // verify values
    std::vector<std::size_t> e(c.size());
    hpx::parallel::detail::sequential_inclusive_scan(
        boost::begin(c), boost::end(c), boost::begin(e), val,
        std::plus<std::size_t>());

    HPX_TEST(std::equal(boost::begin(d), boost::end(d), boost::begin(e)));
}

template <typename IteratorTag>
void test_inclusive_scan2()
{
    using namespace hpx::parallel;

    test_inclusive_scan2(seq, IteratorTag());
    test_inclusive_scan2(par, IteratorTag());
    test_inclusive_scan2(par_vec, IteratorTag());

    test_inclusive_scan2_async(seq(task), IteratorTag());
    test_inclusive_scan2_async(par(task), IteratorTag());

    test_inclusive_scan2(execution_policy(seq), IteratorTag());
    test_inclusive_scan2(execution_policy(par), IteratorTag());
    test_inclusive_scan2(execution_policy(par_vec), IteratorTag());

    test_inclusive_scan2(execution_policy(seq(task)), IteratorTag());
    test_inclusive_scan2(execution_policy(par(task)), IteratorTag());
}

void inclusive_scan_test2()
{
    test_inclusive_scan2<std::random_access_iterator_tag>();
    test_inclusive_scan2<std::forward_iterator_tag>();
    test_inclusive_scan2<std::input_iterator_tag>();
}

///////////////////////////////////////////////////////////////////////////////
template <typename ExPolicy, typename IteratorTag>
void test_inclusive_scan3(ExPolicy const& policy, IteratorTag)
{
    BOOST_STATIC_ASSERT(hpx::parallel::is_execution_policy<ExPolicy>::value);

    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<std::size_t> c(10007);
    std::vector<std::size_t> d(c.size());
    std::fill(boost::begin(c), boost::end(c), std::size_t(1));

    hpx::parallel::inclusive_scan(policy,
        iterator(boost::begin(c)), iterator(boost::end(c)), boost::begin(d));

    // verify values
    std::vector<std::size_t> e(c.size());
    hpx::parallel::detail::sequential_inclusive_scan(
        boost::begin(c), boost::end(c), boost::begin(e), std::size_t(),
        std::plus<std::size_t>());

    HPX_TEST(std::equal(boost::begin(d), boost::end(d), boost::begin(e)));
}

template <typename ExPolicy, typename IteratorTag>
void test_inclusive_scan3_async(ExPolicy const& p, IteratorTag)
{
    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<std::size_t> c(10007);
    std::vector<std::size_t> d(c.size());
    std::fill(boost::begin(c), boost::end(c), std::size_t(1));

    hpx::future<void> f =
        hpx::parallel::inclusive_scan(p,
            iterator(boost::begin(c)), iterator(boost::end(c)), boost::begin(d));
    f.wait();

    // verify values
    std::vector<std::size_t> e(c.size());
    hpx::parallel::detail::sequential_inclusive_scan(
        boost::begin(c), boost::end(c), boost::begin(e), std::size_t(),
        std::plus<std::size_t>());

    HPX_TEST(std::equal(boost::begin(d), boost::end(d), boost::begin(e)));
}

template <typename IteratorTag>
void test_inclusive_scan3()
{
    using namespace hpx::parallel;

    test_inclusive_scan3(seq, IteratorTag());
    test_inclusive_scan3(par, IteratorTag());
    test_inclusive_scan3(par_vec, IteratorTag());

    test_inclusive_scan3_async(seq(task), IteratorTag());
    test_inclusive_scan3_async(par(task), IteratorTag());

    test_inclusive_scan3(execution_policy(seq), IteratorTag());
    test_inclusive_scan3(execution_policy(par), IteratorTag());
    test_inclusive_scan3(execution_policy(par_vec), IteratorTag());

    test_inclusive_scan3(execution_policy(seq(task)), IteratorTag());
    test_inclusive_scan3(execution_policy(par(task)), IteratorTag());
}

void inclusive_scan_test3()
{
    test_inclusive_scan3<std::random_access_iterator_tag>();
    test_inclusive_scan3<std::forward_iterator_tag>();
    test_inclusive_scan3<std::input_iterator_tag>();
}

///////////////////////////////////////////////////////////////////////////////
template <typename ExPolicy, typename IteratorTag>
void test_inclusive_scan_exception(ExPolicy const& policy, IteratorTag)
{
    BOOST_STATIC_ASSERT(hpx::parallel::is_execution_policy<ExPolicy>::value);

    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<std::size_t> c(10007);
    std::vector<std::size_t> d(c.size());
    std::fill(boost::begin(c), boost::end(c), std::size_t(1));

    bool caught_exception = false;
    try {
        hpx::parallel::inclusive_scan(policy,
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
void test_inclusive_scan_exception_async(ExPolicy const& p, IteratorTag)
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
            hpx::parallel::inclusive_scan(p,
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
void test_inclusive_scan_exception()
{
    using namespace hpx::parallel;

    // If the execution policy object is of type vector_execution_policy,
    // std::terminate shall be called. therefore we do not test exceptions
    // with a vector execution policy
    test_inclusive_scan_exception(seq, IteratorTag());
    test_inclusive_scan_exception(par, IteratorTag());

    test_inclusive_scan_exception_async(seq(task), IteratorTag());
    test_inclusive_scan_exception_async(par(task), IteratorTag());

    test_inclusive_scan_exception(execution_policy(seq), IteratorTag());
    test_inclusive_scan_exception(execution_policy(par), IteratorTag());

    test_inclusive_scan_exception(execution_policy(seq(task)), IteratorTag());
    test_inclusive_scan_exception(execution_policy(par(task)), IteratorTag());
}

void inclusive_scan_exception_test()
{
    test_inclusive_scan_exception<std::random_access_iterator_tag>();
    test_inclusive_scan_exception<std::forward_iterator_tag>();
    test_inclusive_scan_exception<std::input_iterator_tag>();
}

///////////////////////////////////////////////////////////////////////////////
template <typename ExPolicy, typename IteratorTag>
void test_inclusive_scan_bad_alloc(ExPolicy const& policy, IteratorTag)
{
    BOOST_STATIC_ASSERT(hpx::parallel::is_execution_policy<ExPolicy>::value);

    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<std::size_t> c(10007);
    std::vector<std::size_t> d(c.size());
    std::fill(boost::begin(c), boost::end(c), std::size_t(1));

    bool caught_exception = false;
    try {
        hpx::parallel::inclusive_scan(policy,
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
void test_inclusive_scan_bad_alloc_async(ExPolicy const& p, IteratorTag)
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
            hpx::parallel::inclusive_scan(p,
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
void test_inclusive_scan_bad_alloc()
{
    using namespace hpx::parallel;

    // If the execution policy object is of type vector_execution_policy,
    // std::terminate shall be called. therefore we do not test exceptions
    // with a vector execution policy
    test_inclusive_scan_bad_alloc(seq, IteratorTag());
    test_inclusive_scan_bad_alloc(par, IteratorTag());

    test_inclusive_scan_bad_alloc_async(seq(task), IteratorTag());
    test_inclusive_scan_bad_alloc_async(par(task), IteratorTag());

    test_inclusive_scan_bad_alloc(execution_policy(seq), IteratorTag());
    test_inclusive_scan_bad_alloc(execution_policy(par), IteratorTag());

    test_inclusive_scan_bad_alloc(execution_policy(seq(task)), IteratorTag());
    test_inclusive_scan_bad_alloc(execution_policy(par(task)), IteratorTag());
}

void inclusive_scan_bad_alloc_test()
{
    test_inclusive_scan_bad_alloc<std::random_access_iterator_tag>();
    test_inclusive_scan_bad_alloc<std::forward_iterator_tag>();
    test_inclusive_scan_bad_alloc<std::input_iterator_tag>();
}

#define FILL_VALUE 10
#define ARRAY_SIZE 10000

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
void test_inclusive_scan_validate(ExPolicy const& p, std::vector<int> &a, std::vector<int> &b)
{
    using namespace hpx::parallel;
    typedef std::vector<int>::iterator Iter;

    // test 1, fill array with numbers counting from 0, then run scan algorithm
    a.clear();
    std::copy(boost::counting_iterator<int>(0), boost::counting_iterator<int>(ARRAY_SIZE), std::back_inserter(a));
    b.resize(a.size());
    hpx::parallel::inclusive_scan(p, a.begin(), a.end(), b.begin(), 0,
                                  [](int bar, int baz){ return bar+baz; });
    //
    for (std::size_t i=0; i<b.size(); ++i) {
        // counting from zero,
        int value = b[i];
        int expected_value  = check_n_triangle(i);
        if (!HPX_TEST(value == expected_value)) break;
    }

    // test 2, fill array with numbers counting from 1, then run scan algorithm
    a.clear();
    std::copy(boost::counting_iterator<int>(1), boost::counting_iterator<int>(ARRAY_SIZE), std::back_inserter(a));
    b.resize(a.size());
    hpx::parallel::inclusive_scan(p, a.begin(), a.end(), b.begin(), 0,
                                  [](int bar, int baz){ return bar+baz; });
    //
    for (std::size_t i=0; i<b.size(); ++i) {
        // counting from 1, use i+1
        int value = b[i];
        int expected_value  = check_n_triangle(i+1);
        if (!HPX_TEST(value == expected_value)) break;
    }

    // test 3, fill array with constant
    a.clear();
    std::fill_n(std::back_inserter(a), ARRAY_SIZE, FILL_VALUE);
    b.resize(a.size());
    hpx::parallel::inclusive_scan(p, a.begin(), a.end(), b.begin(), 0,
                                  [](int bar, int baz){ return bar+baz; });
    //
    for (std::size_t i=0; i<b.size(); ++i) {
        int value = b[i];
        int expected_value  = check_n_const(i+1, FILL_VALUE);
        if (!HPX_TEST(value == expected_value)) break;
    }
}


void inclusive_scan_validate()
{
    std::vector<int> a, b;
    // test scan algorithms using separate array for output
    //  std::cout << " Validating dual arrays " <<std::endl;
    test_inclusive_scan_validate(hpx::parallel::seq, a, b);
    test_inclusive_scan_validate(hpx::parallel::par, a, b);
    // test scan algorithms using same array for input and output
    //  std::cout << " Validating in_place arrays " <<std::endl;
    test_inclusive_scan_validate(hpx::parallel::seq, a, a);
    test_inclusive_scan_validate(hpx::parallel::par, a, a);
}

///////////////////////////////////////////////////////////////////////////////
int hpx_main(boost::program_options::variables_map& vm)
{
    unsigned int seed = (unsigned int)std::time(0);
    if (vm.count("seed"))
        seed = vm["seed"].as<unsigned int>();

    std::cout << "using seed: " << seed << std::endl;
    std::srand(seed);

    inclusive_scan_test1();
    inclusive_scan_test2();
    inclusive_scan_test3();

    inclusive_scan_exception_test();
    inclusive_scan_bad_alloc_test();

    inclusive_scan_validate();

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
