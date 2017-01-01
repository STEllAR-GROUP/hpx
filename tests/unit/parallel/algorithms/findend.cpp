//  Copyright (c) 2014 Grant Mercer
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>
#include <hpx/hpx.hpp>
#include <hpx/include/parallel_find.hpp>
#include <hpx/util/lightweight_test.hpp>

#include <boost/range/functions.hpp>

#include <cstddef>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#include "test_utils.hpp"

////////////////////////////////////////////////////////////////////////////
template <typename ExPolicy, typename IteratorTag>
void test_find_end1(ExPolicy policy, IteratorTag)
{
    static_assert(
        hpx::parallel::execution::is_execution_policy<ExPolicy>::value,
        "hpx::parallel::execution::is_execution_policy<ExPolicy>::value");

    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<std::size_t> c(10007);
    // fill vector with random values above 2
    std::fill(boost::begin(c), boost::end(c), (std::rand() % 100) + 3);
    // create subsequence in middle of vector
    c[c.size()/2] = 1;
    c[c.size()/2 + 1] = 2;

    std::size_t h[] = { 1, 2 };

    iterator index = hpx::parallel::find_end(policy,
        iterator(boost::begin(c)), iterator(boost::end(c)),
        boost::begin(h), boost::end(h));

    base_iterator test_index = boost::begin(c) + c.size()/2;

    HPX_TEST(index == iterator(test_index));
}

template <typename ExPolicy, typename IteratorTag>
void test_find_end1_async(ExPolicy p, IteratorTag)
{
    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<std::size_t> c(10007);
    // fill vector with random values above 2
    std::fill(boost::begin(c), boost::end(c), (std::rand() % 100) + 3);
    // create subsequence in middle of vector
    c[c.size()/2] = 1;
    c[c.size()/2 + 1] = 2;

    std::size_t h[] = { 1, 2 };

    hpx::future<iterator> f =
        hpx::parallel::find_end(p,
            iterator(boost::begin(c)), iterator(boost::end(c)),
            boost::begin(h), boost::end(h));
    f.wait();

    // create iterator at position of value to be found
    base_iterator test_index = boost::begin(c) + c.size()/2;

    HPX_TEST(f.get() == iterator(test_index));
}

template <typename IteratorTag>
void test_find_end1()
{
    using namespace hpx::parallel;
    test_find_end1(execution::seq, IteratorTag());
    test_find_end1(execution::par, IteratorTag());
    test_find_end1(execution::par_unseq, IteratorTag());

    test_find_end1_async(execution::seq(execution::task), IteratorTag());
    test_find_end1_async(execution::par(execution::task), IteratorTag());

#if defined(HPX_HAVE_GENERIC_EXECUTION_POLICY)
    test_find_end1(execution_policy(execution::seq), IteratorTag());
    test_find_end1(execution_policy(execution::par), IteratorTag());
    test_find_end1(execution_policy(execution::par_unseq), IteratorTag());

    test_find_end1(execution_policy(execution::seq(execution::task)), IteratorTag());
    test_find_end1(execution_policy(execution::par(execution::task)), IteratorTag());
#endif
}

void find_end_test1()
{
    test_find_end1<std::random_access_iterator_tag>();
    test_find_end1<std::forward_iterator_tag>();
}

template <typename ExPolicy, typename IteratorTag>
void test_find_end2(ExPolicy policy, IteratorTag)
{
    static_assert(
        hpx::parallel::execution::is_execution_policy<ExPolicy>::value,
        "hpx::parallel::execution::is_execution_policy<ExPolicy>::value");

    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<std::size_t> c(10007);
    // fill vector with random values about 2
    std::fill(boost::begin(c), boost::end(c), (std::rand() % 100) + 3);
    // create subsequence at start and end
    c[0] = 1;
    c[1] = 2;
    c[c.size()-1] = 2;
    c[c.size()-2] = 1;

    std::size_t h[] = { 1, 2 };

    iterator index = hpx::parallel::find_end(policy,
        iterator(boost::begin(c)), iterator(boost::end(c)),
        boost::begin(h), boost::end(h));

    base_iterator test_index = boost::begin(c) + c.size()-2;

    HPX_TEST(index == iterator(test_index));
}

template <typename ExPolicy, typename IteratorTag>
void test_find_end2_async(ExPolicy p, IteratorTag)
{
    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    // fill vector with random values above 2
    std::vector<std::size_t> c(10007);
    std::fill(boost::begin(c), boost::end(c), (std::rand() % 100) + 3);

    // create subsequence at start and end
    c[0] = 1;
    c[1] = 2;
    c[c.size()-1] = 2;
    c[c.size()-2] = 1;

    std::size_t h[] = { 1, 2 };

    hpx::future<iterator> f =
        hpx::parallel::find_end(p,
            iterator(boost::begin(c)), iterator(boost::end(c)),
            boost::begin(h), boost::end(h));
    f.wait();

    // create iterator at position of value to be found
    base_iterator test_index = boost::begin(c) + c.size()-2;

    HPX_TEST(f.get() == iterator(test_index));
}

template <typename IteratorTag>
void test_find_end2()
{
    using namespace hpx::parallel;
    test_find_end2(execution::seq, IteratorTag());
    test_find_end2(execution::par, IteratorTag());
    test_find_end2(execution::par_unseq, IteratorTag());

    test_find_end2_async(execution::seq(execution::task), IteratorTag());
    test_find_end2_async(execution::par(execution::task), IteratorTag());

#if defined(HPX_HAVE_GENERIC_EXECUTION_POLICY)
    test_find_end2(execution_policy(execution::seq), IteratorTag());
    test_find_end2(execution_policy(execution::par), IteratorTag());
    test_find_end2(execution_policy(execution::par_unseq), IteratorTag());

    test_find_end2(execution_policy(execution::seq(execution::task)), IteratorTag());
    test_find_end2(execution_policy(execution::par(execution::task)), IteratorTag());
#endif
}

void find_end_test2()
{
    test_find_end2<std::random_access_iterator_tag>();
    test_find_end2<std::forward_iterator_tag>();
}

template <typename ExPolicy, typename IteratorTag>
void test_find_end3(ExPolicy policy, IteratorTag)
{
    static_assert(
        hpx::parallel::execution::is_execution_policy<ExPolicy>::value,
        "hpx::parallel::execution::is_execution_policy<ExPolicy>::value");

    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    // fill vector with random values above 2
    std::vector<std::size_t> c(10007);
    std::fill(boost::begin(c), boost::end(c), (std::rand() % 100) + 3);

    // create subsequence large enough to always be split into multiple partitions
    std::iota(boost::begin(c), boost::begin(c) + c.size()/16+1, 1);
    std::size_t sub_size = c.size()/16 + 1;

    std::vector<std::size_t> h(sub_size);
    std::iota(boost::begin(h), boost::end(h), 1);

    iterator index = hpx::parallel::find_end(policy,
        iterator(boost::begin(c)), iterator(boost::end(c)),
        boost::begin(h), boost::end(h));

    base_iterator test_index = boost::begin(c);

    HPX_TEST(index == iterator(test_index));
}

template <typename ExPolicy, typename IteratorTag>
void test_find_end3_async(ExPolicy p, IteratorTag)
{
    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    // fill vector with random values above 6
    std::vector<std::size_t> c(10007);
    std::fill(boost::begin(c), boost::end(c), (std::rand() % 100) + 7);

    // create subsequence large enough to always be split into multiple partitions
    std::iota(boost::begin(c), boost::begin(c) + c.size()/16+1, 1);
    std::size_t sub_size = c.size()/16 + 1;

    std::vector<std::size_t> h(sub_size);
    std::iota(boost::begin(h), boost::end(h), 1);

    // create only two partitions, splitting the desired sub sequence into
    // separate partitions.
    hpx::future<iterator> f =
        hpx::parallel::find_end(p,
            iterator(boost::begin(c)), iterator(boost::end(c)),
            boost::begin(h), boost::end(h));
    f.wait();

    //create iterator at position of value to be found
    base_iterator test_index = boost::begin(c);

    HPX_TEST(f.get() == iterator(test_index));
}

template <typename IteratorTag>
void test_find_end3()
{
    using namespace hpx::parallel;
    test_find_end3(execution::seq, IteratorTag());
    test_find_end3(execution::par, IteratorTag());
    test_find_end3(execution::par_unseq, IteratorTag());

    test_find_end3_async(execution::seq(execution::task), IteratorTag());
    test_find_end3_async(execution::par(execution::task), IteratorTag());

#if defined(HPX_HAVE_GENERIC_EXECUTION_POLICY)
    test_find_end3(execution_policy(execution::seq), IteratorTag());
    test_find_end3(execution_policy(execution::par), IteratorTag());
    test_find_end3(execution_policy(execution::par_unseq), IteratorTag());

    test_find_end3(execution_policy(execution::seq(execution::task)), IteratorTag());
    test_find_end3(execution_policy(execution::par(execution::task)), IteratorTag());
#endif
}

void find_end_test3()
{
    test_find_end3<std::random_access_iterator_tag>();
    test_find_end3<std::forward_iterator_tag>();
}

template <typename ExPolicy, typename IteratorTag>
void test_find_end4(ExPolicy policy, IteratorTag)
{
    static_assert(
        hpx::parallel::execution::is_execution_policy<ExPolicy>::value,
        "hpx::parallel::execution::is_execution_policy<ExPolicy>::value");

    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<std::size_t> c(10007);
    // fill vector with random values above 2
    std::fill(boost::begin(c), boost::end(c), (std::rand() % 100) + 3);
    // create subsequence in middle of vector
    c[c.size()/2] = 1;
    c[c.size()/2 + 1] = 2;

    std::size_t h[] = { 1, 2 };

    iterator index = hpx::parallel::find_end(policy,
        iterator(boost::begin(c)), iterator(boost::end(c)),
        boost::begin(h), boost::end(h),
        [](std::size_t v1, std::size_t v2) {
            return !(v1 != v2);
        });

    base_iterator test_index = boost::begin(c) + c.size()/2;

    HPX_TEST(index == iterator(test_index));
}

template <typename ExPolicy, typename IteratorTag>
void test_find_end4_async(ExPolicy p, IteratorTag)
{
    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    // fill vector with random values above 2
    std::vector<std::size_t> c(10007);
    std::fill(boost::begin(c), boost::end(c), (std::rand() % 100) + 3);

    // create subsequence in middle of vector
    c[c.size()/2] = 1;
    c[c.size()/2 + 1] = 2;

    std::size_t h[] = { 1, 2 };

    hpx::future<iterator> f =
        hpx::parallel::find_end(p,
            iterator(boost::begin(c)), iterator(boost::end(c)),
            boost::begin(h), boost::end(h),
            [](std::size_t v1, std::size_t v2) {
                return !(v1 != v2);
        });
    f.wait();

    //create iterator at position of value to be found
    base_iterator test_index = boost::begin(c) + c.size()/2;

    HPX_TEST(f.get() == iterator(test_index));
}

template <typename IteratorTag>
void test_find_end4()
{
    using namespace hpx::parallel;
    test_find_end4(execution::seq, IteratorTag());
    test_find_end4(execution::par, IteratorTag());
    test_find_end4(execution::par_unseq, IteratorTag());

    test_find_end4_async(execution::seq(execution::task), IteratorTag());
    test_find_end4_async(execution::par(execution::task), IteratorTag());

#if defined(HPX_HAVE_GENERIC_EXECUTION_POLICY)
    test_find_end4(execution_policy(execution::seq), IteratorTag());
    test_find_end4(execution_policy(execution::par), IteratorTag());
    test_find_end4(execution_policy(execution::par_unseq), IteratorTag());

    test_find_end4(execution_policy(execution::seq(execution::task)), IteratorTag());
    test_find_end4(execution_policy(execution::par(execution::task)), IteratorTag());
#endif
}

void find_end_test4()
{
    test_find_end4<std::random_access_iterator_tag>();
    test_find_end4<std::forward_iterator_tag>();
}

///////////////////////////////////////////////////////////////////////////////
template <typename ExPolicy, typename IteratorTag>
void test_find_end_exception(ExPolicy policy, IteratorTag)
{
    static_assert(
        hpx::parallel::execution::is_execution_policy<ExPolicy>::value,
        "hpx::parallel::execution::is_execution_policy<ExPolicy>::value");

    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::decorated_iterator<base_iterator, IteratorTag>
        decorated_iterator;
    std::vector<std::size_t> c(10007);
    std::iota(boost::begin(c), boost::end(c), std::rand() + 1);
    c[c.size()/2] = 1;
    c[c.size()/2+1] = 2;

    std::vector<std::size_t> h;
    h.push_back(1);
    h.push_back(2);

    bool caught_exception = false;
    try {
        hpx::parallel::find_end(policy,
            decorated_iterator(
                boost::begin(c),
                [](){ throw std::runtime_error("test"); }),
            decorated_iterator(
                boost::end(c),
                [](){ throw std::runtime_error("test"); }),
            boost::begin(h), boost::end(h));
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
void test_find_end_exception_async(ExPolicy p, IteratorTag)
{
    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::decorated_iterator<base_iterator, IteratorTag>
        decorated_iterator;

    std::vector<std::size_t> c(10007);
    std::iota(boost::begin(c), boost::end(c), std::rand() + 1);
    c[c.size()/2] = 1;
    c[c.size()/2+1] = 2;

    std::size_t h[] = { 1, 2 };

    bool caught_exception = false;
    bool returned_from_algorithm = false;
    try {
        hpx::future<decorated_iterator> f =
            hpx::parallel::find_end(p,
                decorated_iterator(
                    boost::begin(c),
                    [](){ throw std::runtime_error("test"); }),
                decorated_iterator(
                    boost::end(c),
                    [](){ throw std::runtime_error("test"); }),
            boost::begin(h), boost::end(h));
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
void test_find_end_exception()
{
    using namespace hpx::parallel;

    // If the execution policy object is of type vector_execution_policy,
    // std::terminate shall be called. therefore we do not test exceptions
    // with a vector execution policy
    test_find_end_exception(execution::seq, IteratorTag());
    test_find_end_exception(execution::par, IteratorTag());

    test_find_end_exception_async(execution::seq(execution::task), IteratorTag());
    test_find_end_exception_async(execution::par(execution::task), IteratorTag());

#if defined(HPX_HAVE_GENERIC_EXECUTION_POLICY)
    test_find_end_exception(execution_policy(execution::seq), IteratorTag());

    test_find_end_exception(execution_policy(execution::seq(execution::task)),
        IteratorTag());
    test_find_end_exception(execution_policy(execution::par(execution::task)),
        IteratorTag());
#endif
}

void find_end_exception_test()
{
    test_find_end_exception<std::random_access_iterator_tag>();
    test_find_end_exception<std::forward_iterator_tag>();
}

//////////////////////////////////////////////////////////////////////////////
template <typename ExPolicy, typename IteratorTag>
void test_find_end_bad_alloc(ExPolicy policy, IteratorTag)
{
    static_assert(
        hpx::parallel::execution::is_execution_policy<ExPolicy>::value,
        "hpx::parallel::execution::is_execution_policy<ExPolicy>::value");

    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::decorated_iterator<base_iterator, IteratorTag>
        decorated_iterator;

    std::vector<std::size_t> c(100007);
    std::iota(boost::begin(c), boost::end(c), std::rand() + 1);
    c[c.size()/2] = 0;

    std::size_t h[] = { 1, 2 };

    bool caught_bad_alloc = false;
    try {
        hpx::parallel::find_end(policy,
            decorated_iterator(
                boost::begin(c),
                [](){ throw std::bad_alloc(); }),
            decorated_iterator(
                boost::end(c),
                [](){ throw std::bad_alloc(); }),
            boost::begin(h), boost::end(h));
        HPX_TEST(false);
    }
    catch(std::bad_alloc const&) {
        caught_bad_alloc = true;
    }
    catch(...) {
        HPX_TEST(false);
    }

    HPX_TEST(caught_bad_alloc);
}

template <typename ExPolicy, typename IteratorTag>
void test_find_end_bad_alloc_async(ExPolicy p, IteratorTag)
{
    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::decorated_iterator<base_iterator, IteratorTag>
        decorated_iterator;

    std::vector<std::size_t> c(10007);
    std::iota(boost::begin(c), boost::end(c), std::rand() + 1);
    c[c.size()/2] = 0;

    std::size_t h[] = { 1, 2 };

    bool caught_bad_alloc = false;
    bool returned_from_algorithm = false;
    try {
        hpx::future<decorated_iterator> f =
            hpx::parallel::find_end(p,
                decorated_iterator(
                    boost::begin(c),
                    [](){ throw std::bad_alloc(); }),
                decorated_iterator(
                    boost::end(c),
                    [](){ throw std::bad_alloc(); }),
                boost::begin(h), boost::end(h));
        returned_from_algorithm = true;
        f.get();

        HPX_TEST(false);
    }
    catch(std::bad_alloc const&) {
        caught_bad_alloc = true;
    }
    catch(...) {
        HPX_TEST(false);
    }

    HPX_TEST(caught_bad_alloc);
    HPX_TEST(returned_from_algorithm);
}

template <typename IteratorTag>
void test_find_end_bad_alloc()
{
    using namespace hpx::parallel;

    // If the execution policy object is of type vector_execution_policy,
    // std::terminate shall be called. therefore we do not test exceptions
    // with a vector execution policy
    test_find_end_bad_alloc(execution::seq, IteratorTag());
    test_find_end_bad_alloc(execution::par, IteratorTag());

    test_find_end_bad_alloc_async(execution::seq(execution::task), IteratorTag());
    test_find_end_bad_alloc_async(execution::par(execution::task), IteratorTag());

#if defined(HPX_HAVE_GENERIC_EXECUTION_POLICY)
    test_find_end_bad_alloc(execution_policy(execution::seq), IteratorTag());
    test_find_end_bad_alloc(execution_policy(execution::par), IteratorTag());

    test_find_end_bad_alloc(execution_policy(execution::seq(execution::task)),
        IteratorTag());
    test_find_end_bad_alloc(execution_policy(execution::par(execution::task)),
        IteratorTag());
#endif
}

void find_end_bad_alloc_test()
{
    test_find_end_bad_alloc<std::random_access_iterator_tag>();
    test_find_end_bad_alloc<std::forward_iterator_tag>();
}

int hpx_main(boost::program_options::variables_map& vm)
{
    unsigned int seed = (unsigned int)std::time(nullptr);
    if (vm.count("seed"))
        seed = vm["seed"].as<unsigned int>();

    std::cout << "using seed: " << seed << std::endl;
    std::srand(seed);

    find_end_test1();
    find_end_test2();
    find_end_test3();
    find_end_test4();
    find_end_exception_test();
    find_end_bad_alloc_test();
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
    std::vector<std::string> const cfg = {
        "hpx.os_threads=all"
    };

    // Initialize and run HPX
    HPX_TEST_EQ_MSG(hpx::init(desc_commandline, argc, argv, cfg), 0,
        "HPX main exited with non-zero status");

    return hpx::util::report_errors();
}
