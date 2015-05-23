//  Copyright (c) 2015 Daniel Bourgeois
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>
#include <hpx/hpx.hpp>
#include <hpx/include/parallel_is_sorted.hpp>
#include <hpx/util/lightweight_test.hpp>

#include <boost/range/functions.hpp>

#include "test_utils.hpp"

#include <iostream>

////////////////////////////////////////////////////////////////////////////////
template <typename ExPolicy, typename IteratorTag>
void test_sorted1(ExPolicy const& policy, IteratorTag)
{
    BOOST_STATIC_ASSERT(hpx::parallel::is_execution_policy<ExPolicy>::value);

    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<std::size_t> c(10007);
    //Fill with sorted values from 0 to 10006
    std::iota(boost::begin(c), boost::end(c), 0);

    bool is_ordered = hpx::parallel::is_sorted(policy,
        iterator(boost::begin(c)), iterator(boost::end(c)));

    HPX_TEST(is_ordered);
}

template <typename ExPolicy, typename IteratorTag>
void test_sorted1_async(ExPolicy const& p, IteratorTag)
{
    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<std::size_t> c(10007);
    //Fill with sorted values from 0 to 10006
    std::iota(boost::begin(c), boost::end(c), 0);

    hpx::future<bool> f =
        hpx::parallel::is_sorted(p,
        iterator(boost::begin(c)), iterator(boost::end(c)));
    f.wait();

    HPX_TEST(f.get());
}

template <typename IteratorTag>
void test_sorted1()
{
    using namespace hpx::parallel;
    test_sorted1(seq, IteratorTag());
    test_sorted1(par, IteratorTag());
    test_sorted1(par_vec, IteratorTag());

    test_sorted1_async(seq(task), IteratorTag());
    test_sorted1_async(par(task), IteratorTag());


    test_sorted1(execution_policy(seq), IteratorTag());
    test_sorted1(execution_policy(par), IteratorTag());
    test_sorted1(execution_policy(par_vec), IteratorTag());
    test_sorted1(execution_policy(seq(task)), IteratorTag());
    test_sorted1(execution_policy(par(task)), IteratorTag());
}

void sorted_test1()
{
    test_sorted1<std::random_access_iterator_tag>();
    test_sorted1<std::forward_iterator_tag>();
}

////////////////////////////////////////////////////////////////////////////////
template <typename ExPolicy, typename IteratorTag>
void test_sorted2(ExPolicy const& policy, IteratorTag)
{
    BOOST_STATIC_ASSERT(hpx::parallel::is_execution_policy<ExPolicy>::value);

    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<std::size_t> c(10007);
    //Fill with sorted values from 0 to 10006
    std::iota(boost::begin(c), boost::end(c), 0);
    //Add a certain large value in middle of array to ignore
    std::size_t ignore = 20000;
    c[c.size()/2] = ignore;
    //Provide custom predicate to ignore the value of ignore
    //pred should return true when it is given something deemed not sorted
    auto pred = [&ignore](std::size_t ahead, std::size_t behind)
    {
        return behind > ahead && behind != ignore;
    };

    bool is_ordered = hpx::parallel::is_sorted(policy,
        iterator(boost::begin(c)), iterator(boost::end(c)), pred);

    HPX_TEST(is_ordered);
}

template <typename ExPolicy, typename IteratorTag>
void test_sorted2_async(ExPolicy const& p, IteratorTag)
{
    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<std::size_t> c(10007);
    //Fill with sorted values from 0 to 10006
    std::iota(boost::begin(c), boost::end(c), 0);
    //Add a certain large value in middle of array to ignore
    std::size_t ignore = 20000;
    c[c.size()/2] = ignore;
    //Provide custom predicate to ignore the value of ignore
    //pred should return true when it is given something deemed not sorted
    auto pred = [&ignore](std::size_t ahead, std::size_t behind)
    {
        return behind > ahead && behind != ignore;
    };

    hpx::future<bool> f = hpx::parallel::is_sorted(p,
        iterator(boost::begin(c)), iterator(boost::end(c)), pred);
    f.wait();

    HPX_TEST(f.get());
}

template <typename IteratorTag>
void test_sorted2()
{
    using namespace hpx::parallel;
    test_sorted2(seq, IteratorTag());
    test_sorted2(par, IteratorTag());
    test_sorted2(par_vec, IteratorTag());

    test_sorted2_async(seq(task), IteratorTag());
    test_sorted2_async(par(task), IteratorTag());


    test_sorted2(execution_policy(seq), IteratorTag());
    test_sorted2(execution_policy(par), IteratorTag());
    test_sorted2(execution_policy(par_vec), IteratorTag());
    test_sorted2(execution_policy(seq(task)), IteratorTag());
    test_sorted2(execution_policy(par(task)), IteratorTag());
}

void sorted_test2()
{
    test_sorted2<std::random_access_iterator_tag>();
    test_sorted2<std::forward_iterator_tag>();
}

////////////////////////////////////////////////////////////////////////////////
template <typename ExPolicy, typename IteratorTag>
void test_sorted3(ExPolicy const& policy, IteratorTag)
{
    BOOST_STATIC_ASSERT(hpx::parallel::is_execution_policy<ExPolicy>::value);

    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<std::size_t> c_beg(10007);
    std::vector<std::size_t> c_end(10007);
    //Fill with sorted values from 0 to 10006
    std::iota(boost::begin(c_beg), boost::end(c_beg), 0);
    std::iota(boost::begin(c_end), boost::end(c_end), 0);
    //add unsorted element to c_beg, c_end at the beginning, end respectively
    c_beg[0] = 20000;
    c_end[c_end.size()-1] = 0;

    bool is_ordered1 = hpx::parallel::is_sorted(policy,
        iterator(boost::begin(c_beg)), iterator(boost::end(c_beg)));
    bool is_ordered2 = hpx::parallel::is_sorted(policy,
        iterator(boost::begin(c_end)), iterator(boost::end(c_end)));

    HPX_TEST(!is_ordered1);
    HPX_TEST(!is_ordered2);
}

template <typename ExPolicy, typename IteratorTag>
void test_sorted3_async(ExPolicy const& p, IteratorTag)
{
    BOOST_STATIC_ASSERT(hpx::parallel::is_execution_policy<ExPolicy>::value);

    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<std::size_t> c_beg(10007);
    std::vector<std::size_t> c_end(10007);
    //Fill with sorted values from 0 to 10006
    std::iota(boost::begin(c_beg), boost::end(c_beg), 0);
    std::iota(boost::begin(c_end), boost::end(c_end), 0);
    //add unsorted element to c_beg, c_end at the beginning, end respectively
    c_beg[0] = 20000;
    c_end[c_end.size()-1] = 0;

    hpx::future<bool> f1 = hpx::parallel::is_sorted(p,
        iterator(boost::begin(c_beg)), iterator(boost::end(c_beg)));
    hpx::future<bool> f2 = hpx::parallel::is_sorted(p,
        iterator(boost::begin(c_end)), iterator(boost::end(c_end)));
    f1.wait();
    HPX_TEST(!f1.get());
    f2.wait();
    HPX_TEST(!f2.get());
}

template <typename IteratorTag>
void test_sorted3()
{
    using namespace hpx::parallel;
    test_sorted3(seq, IteratorTag());
    test_sorted3(par, IteratorTag());
    test_sorted3(par_vec, IteratorTag());

    test_sorted3_async(seq(task), IteratorTag());
    test_sorted3_async(par(task), IteratorTag());


    test_sorted3(execution_policy(seq), IteratorTag());
    test_sorted3(execution_policy(par), IteratorTag());
    test_sorted3(execution_policy(par_vec), IteratorTag());
    test_sorted3(execution_policy(seq(task)), IteratorTag());
    test_sorted3(execution_policy(par(task)), IteratorTag());
}

void sorted_test3()
{
    test_sorted3<std::random_access_iterator_tag>();
    test_sorted3<std::forward_iterator_tag>();
}

////////////////////////////////////////////////////////////////////////////////
template <typename ExPolicy, typename IteratorTag>
void test_sorted_exception(ExPolicy const& policy, IteratorTag)
{
    BOOST_STATIC_ASSERT(hpx::parallel::is_execution_policy<ExPolicy>::value);

    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::decorated_iterator<base_iterator, IteratorTag>
        decorated_iterator;
    std::vector<std::size_t> c(10007);
    std::iota(boost::begin(c), boost::end(c), 0);

    bool caught_exception = false;
    try{
        hpx::parallel::is_sorted(policy,
            decorated_iterator(
                boost::begin(c),
                [](){ throw std::runtime_error("test"); }),
            decorated_iterator(
                boost::end(c),
                [](){ throw std::runtime_error("test"); }));
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
void test_sorted_async_exception(ExPolicy const& p, IteratorTag)
{
    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::decorated_iterator<base_iterator, IteratorTag>
        decorated_iterator;

    std::vector<std::size_t> c(10007);
    std::iota(boost::begin(c), boost::end(c), std::rand() + 1);

    bool caught_exception = false;
    try {
        hpx::future<bool> f =
            hpx::parallel::is_sorted(p,
                decorated_iterator(
                    boost::begin(c),
                    [](){ throw std::runtime_error("test"); }),
                decorated_iterator(
                    boost::end(c),
                    [](){ throw std::runtime_error("test"); }));
        f.get();

        HPX_TEST(false);
    }
    catch(hpx::exception_list const& e) {
        caught_exception = true;
        test::test_num_exceptions<
            ExPolicy, IteratorTag
            >::call(p, e);
    }
    catch(...) {
        HPX_TEST(false);
    }

    HPX_TEST(caught_exception);
}

template <typename IteratorTag>
void test_sorted_exception()
{
    using namespace hpx::parallel;
    //If the execution policy object is of type vector_execution_policy,
    //  std::terminate shall be called. Therefore we do not test exceptions
    //  with a vector execution policy
    test_sorted_exception(seq, IteratorTag());
    test_sorted_exception(par, IteratorTag());

    test_sorted_async_exception(seq(task), IteratorTag());
    test_sorted_async_exception(par(task), IteratorTag());

    test_sorted_exception(execution_policy(par), IteratorTag());
    test_sorted_exception(execution_policy(seq(task)), IteratorTag());
    test_sorted_exception(execution_policy(par(task)), IteratorTag());
}
void sorted_exception_test()
{
    test_sorted_exception<std::random_access_iterator_tag>();
    test_sorted_exception<std::forward_iterator_tag>();
}

////////////////////////////////////////////////////////////////////////////////
template <typename ExPolicy, typename IteratorTag>
void test_sorted_bad_alloc(ExPolicy const& policy, IteratorTag)
{
    BOOST_STATIC_ASSERT(hpx::parallel::is_execution_policy<ExPolicy>::value);

    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::decorated_iterator<base_iterator, IteratorTag>
        decorated_iterator;

    std::vector<std::size_t> c(100007);
    std::iota(boost::begin(c), boost::end(c), 0);

    bool caught_bad_alloc = false;
    try {
        hpx::parallel::is_sorted(policy,
            decorated_iterator(
                boost::begin(c),
                [](){ throw std::bad_alloc(); }),
            decorated_iterator(
                boost::end(c),
                [](){ throw std::bad_alloc(); }));
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
void test_sorted_async_bad_alloc(ExPolicy const& p, IteratorTag)
{
    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::decorated_iterator<base_iterator, IteratorTag>
        decorated_iterator;

    std::vector<std::size_t> c(10007);
    std::iota(boost::begin(c), boost::end(c), 0);

    bool caught_bad_alloc = false;
    try {
        hpx::future<bool> f =
            hpx::parallel::is_sorted(p,
                decorated_iterator(
                    boost::begin(c),
                    [](){ throw std::bad_alloc(); }),
                decorated_iterator(
                    boost::end(c),
                    [](){ throw std::bad_alloc(); }));

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
}

template <typename IteratorTag>
void test_sorted_bad_alloc()
{
    using namespace hpx::parallel;

    // If the execution policy object is of type vector_execution_policy,
    // std::terminate shall be called. therefore we do not test exceptions
    // with a vector execution policy
    test_sorted_bad_alloc(par, IteratorTag());
    test_sorted_bad_alloc(seq, IteratorTag());

    test_sorted_async_bad_alloc(seq(task), IteratorTag());
    test_sorted_async_bad_alloc(par(task), IteratorTag());

    test_sorted_bad_alloc(execution_policy(par), IteratorTag());
    test_sorted_bad_alloc(execution_policy(seq), IteratorTag());
    test_sorted_bad_alloc(execution_policy(seq(task)), IteratorTag());
    test_sorted_bad_alloc(execution_policy(par(task)), IteratorTag());
}

void sorted_bad_alloc_test()
{
    test_sorted_bad_alloc<std::random_access_iterator_tag>();
    test_sorted_bad_alloc<std::forward_iterator_tag>();
}

int hpx_main(boost::program_options::variables_map& vm)
{
    sorted_test1();
    sorted_test2();
    sorted_test3();
    sorted_exception_test();
    sorted_bad_alloc_test();
    return hpx::finalize();
}

int main(int argc, char* argv[])
{
    using namespace boost::program_options;
    options_description desc_commandline(
        "Usage: " HPX_APPLICATION_STRING " [options]");

    std::vector<std::string> cfg;
    cfg.push_back("hpx.os_threads=" +
        boost::lexical_cast<std::string>(hpx::threads::hardware_concurrency()));

    HPX_TEST_EQ_MSG(hpx::init(desc_commandline, argc, argv, cfg), 0,
        "HPX main exited with non-zero status");

    return hpx::util::report_errors();
}
