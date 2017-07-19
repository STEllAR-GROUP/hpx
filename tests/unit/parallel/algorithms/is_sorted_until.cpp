//  Copyright (c) 2015 Daniel Bourgeois
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>
#include <hpx/hpx.hpp>
#include <hpx/include/parallel_is_sorted.hpp>
#include <hpx/util/lightweight_test.hpp>

#include <cstddef>
#include <iostream>
#include <iterator>
#include <numeric>
#include <string>
#include <vector>

#include "test_utils.hpp"

////////////////////////////////////////////////////////////////////////////////
template <typename ExPolicy, typename IteratorTag>
void test_sorted_until1(ExPolicy policy, IteratorTag)
{
    static_assert(
        hpx::parallel::execution::is_execution_policy<ExPolicy>::value,
        "hpx::parallel::execution::is_execution_policy<ExPolicy>::value");

    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<std::size_t> c(10007);
    std::iota(std::begin(c), std::end(c), 0);

    iterator until = hpx::parallel::is_sorted_until(policy,
        iterator(std::begin(c)), iterator(std::end(c)));

    base_iterator test_index = std::end(c);

    HPX_TEST(until == iterator(test_index));
}

template <typename ExPolicy, typename IteratorTag>
void test_sorted_until1_async(ExPolicy p, IteratorTag)
{
    static_assert(
        hpx::parallel::execution::is_execution_policy<ExPolicy>::value,
        "hpx::parallel::execution::is_execution_policy<ExPolicy>::value");

    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<std::size_t> c(10007);
    std::iota(std::begin(c), std::end(c), 0);

    hpx::future<iterator> f = hpx::parallel::is_sorted_until(p,
        iterator(std::begin(c)), iterator(std::end(c)));

    base_iterator test_index = std::end(c);

    f.wait();
    HPX_TEST(f.get() == iterator(test_index));

}

template <typename IteratorTag>
void test_sorted_until1()
{
    using namespace hpx::parallel;
    test_sorted_until1(execution::seq, IteratorTag());
        //calls sequential and gets iter
    test_sorted_until1(execution::par, IteratorTag());
        //calls parallel and gets iter
    test_sorted_until1(execution::par_unseq, IteratorTag());
        //calls parallel and gets iter
    test_sorted_until1_async(execution::seq(execution::task), IteratorTag());
        //calls sequential and gets future
    test_sorted_until1_async(execution::par(execution::task), IteratorTag());
        //calls parallel and gets future

#if defined(HPX_HAVE_GENERIC_EXECUTION_POLICY)
    test_sorted_until1(execution_policy(execution::seq), IteratorTag());
    //calls sequential and gets iter
    test_sorted_until1(execution_policy(execution::par), IteratorTag());
    //calls parallel and gets iter
    test_sorted_until1(execution_policy(execution::par_unseq), IteratorTag());
    //calls parallel and gets iter
    test_sorted_until1(execution_policy(execution::seq(execution::task)),
        IteratorTag());
    //calls sequential and gets iter
    test_sorted_until1(execution_policy(execution::par(execution::task)),
        IteratorTag());
    //calls parallel and gets iter
#endif
}

void sorted_until_test1()
{
    test_sorted_until1<std::random_access_iterator_tag>();
    test_sorted_until1<std::forward_iterator_tag>();
}

////////////////////////////////////////////////////////////////////////////////
template <typename ExPolicy, typename IteratorTag>
void test_sorted_until2(ExPolicy policy, IteratorTag)
{
    static_assert(
        hpx::parallel::execution::is_execution_policy<ExPolicy>::value,
        "hpx::parallel::execution::is_execution_policy<ExPolicy>::value");

    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<std::size_t> c(10007);
    //Fill with sorted values from 0 to 10006
    std::iota(std::begin(c), std::end(c), 0);
    //Add a certain large value in middle of array to ignore
    std::size_t ignore = 20000;
    c[c.size()/2] = ignore;
    //Provide custom predicate to ignore the value of ignore
    //pred should return true when it is given something deemed not sorted
    auto pred = [&ignore](std::size_t ahead, std::size_t behind)
    {
        return behind > ahead && behind != ignore;
    };

    iterator until = hpx::parallel::is_sorted_until(policy,
        iterator(std::begin(c)), iterator(std::end(c)), pred);

    base_iterator test_index = std::end(c);

    HPX_TEST(until == iterator(test_index));
}

template <typename ExPolicy, typename IteratorTag>
void test_sorted_until2_async(ExPolicy p, IteratorTag)
{
    static_assert(
        hpx::parallel::execution::is_execution_policy<ExPolicy>::value,
        "hpx::parallel::execution::is_execution_policy<ExPolicy>::value");

    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<std::size_t> c(10007);
    //Fill with sorted values from 0 to 10006
    std::iota(std::begin(c), std::end(c), 0);
    //Add a certain large value in middle of array to ignore
    std::size_t ignore = 20000;
    c[c.size()/2] = ignore;
    //Provide custom predicate to ignore the value of ignore
    //pred should return true when it is given something deemed not sorted
    auto pred = [&ignore](std::size_t ahead, std::size_t behind)
    {
        return behind > ahead && behind != ignore;
    };

    hpx::future<iterator> f = hpx::parallel::is_sorted_until(p,
        iterator(std::begin(c)), iterator(std::end(c)), pred);

    base_iterator test_index = std::end(c);
    f.wait();
    HPX_TEST(f.get() == iterator(test_index));
}

template <typename IteratorTag>
void test_sorted_until2()
{
    using namespace hpx::parallel;
    test_sorted_until2(execution::seq, IteratorTag());
    test_sorted_until2(execution::par, IteratorTag());
    test_sorted_until2(execution::par_unseq, IteratorTag());

    test_sorted_until2_async(execution::seq(execution::task), IteratorTag());
    test_sorted_until2_async(execution::par(execution::task), IteratorTag());

#if defined(HPX_HAVE_GENERIC_EXECUTION_POLICY)
    test_sorted_until2(execution_policy(execution::seq), IteratorTag());
    test_sorted_until2(execution_policy(execution::par), IteratorTag());
    test_sorted_until2(execution_policy(execution::par_unseq), IteratorTag());
    test_sorted_until2(execution_policy(execution::seq(execution::task)),
        IteratorTag());
    test_sorted_until2(execution_policy(execution::par(execution::task)),
        IteratorTag());
#endif
}

void sorted_until_test2()
{
    test_sorted_until2<std::random_access_iterator_tag>();
    test_sorted_until2<std::forward_iterator_tag>();
}

////////////////////////////////////////////////////////////////////////////////
template <typename ExPolicy, typename IteratorTag>
void test_sorted_until3(ExPolicy policy, IteratorTag)
{
    static_assert(
        hpx::parallel::execution::is_execution_policy<ExPolicy>::value,
        "hpx::parallel::execution::is_execution_policy<ExPolicy>::value");

    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;
    //test the following:
    // put unsorted elements at each ends
    // put two unsorted elements in the middle

    std::vector<std::size_t> c1(10007);
    std::vector<std::size_t> c2(10007);
    std::iota(std::begin(c1), std::end(c1), 0);
    std::iota(std::begin(c2), std::end(c2), 0);
    c1[0] = 20000;
    c1[c1.size() - 1] = 0;
    c2[c2.size()/3] = 0;
    c2[2*c2.size()/3] = 0;

    iterator until1 = hpx::parallel::is_sorted_until(policy,
        iterator(std::begin(c1)), iterator(std::end(c1)));
    iterator until2 = hpx::parallel::is_sorted_until(policy,
        iterator(std::begin(c2)), iterator(std::end(c2)));

    base_iterator test_index1 = std::begin(c1) + 1;
    base_iterator test_index2 = std::begin(c2) + c2.size()/3;

    HPX_TEST(until1 == iterator(test_index1));
    HPX_TEST(until2 == iterator(test_index2));

}

template <typename ExPolicy, typename IteratorTag>
void test_sorted_until3_async(ExPolicy p, IteratorTag)
{
    static_assert(
        hpx::parallel::execution::is_execution_policy<ExPolicy>::value,
        "hpx::parallel::execution::is_execution_policy<ExPolicy>::value");

    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;
    //test the following:
    // put unsorted elements at each ends
    // put two unsorted elements in the middle

    std::vector<std::size_t> c1(10007);
    std::vector<std::size_t> c2(10007);
    std::iota(std::begin(c1), std::end(c1), 0);
    std::iota(std::begin(c2), std::end(c2), 0);
    c1[0] = 20000;
    c1[c1.size() - 1] = 0;
    c2[c2.size()/3] = 0;
    c2[2*c2.size()/3] = 0;

    hpx::future<iterator> f1 = hpx::parallel::is_sorted_until(p,
        iterator(std::begin(c1)), iterator(std::end(c1)));
    hpx::future<iterator> f2 = hpx::parallel::is_sorted_until(p,
        iterator(std::begin(c2)), iterator(std::end(c2)));

    base_iterator test_index1 = std::begin(c1) + 1;
    base_iterator test_index2 = std::begin(c2) + c2.size()/3;

    f1.wait();
    HPX_TEST(f1.get() == iterator(test_index1));
    f2.wait();
    HPX_TEST(f2.get() == iterator(test_index2));
}

template <typename IteratorTag>
void test_sorted_until3()
{
    using namespace hpx::parallel;
    test_sorted_until3(execution::seq, IteratorTag());
    test_sorted_until3(execution::par, IteratorTag());
    test_sorted_until3(execution::par_unseq, IteratorTag());

    test_sorted_until3_async(execution::seq(execution::task), IteratorTag());
    test_sorted_until3_async(execution::par(execution::task), IteratorTag());

#if defined(HPX_HAVE_GENERIC_EXECUTION_POLICY)
    test_sorted_until3(execution_policy(execution::seq), IteratorTag());
    test_sorted_until3(execution_policy(execution::par), IteratorTag());
    test_sorted_until3(execution_policy(execution::par_unseq), IteratorTag());
    test_sorted_until3(execution_policy(execution::seq(execution::task)),
        IteratorTag());
    test_sorted_until3(execution_policy(execution::par(execution::task)),
        IteratorTag());
#endif
}

void sorted_until_test3()
{
    test_sorted_until3<std::random_access_iterator_tag>();
    test_sorted_until3<std::forward_iterator_tag>();
}

////////////////////////////////////////////////////////////////////////////////
template <typename ExPolicy, typename IteratorTag>
void test_sorted_until_exception(ExPolicy policy, IteratorTag)
{
    static_assert(
        hpx::parallel::execution::is_execution_policy<ExPolicy>::value,
        "hpx::parallel::execution::is_execution_policy<ExPolicy>::value");

    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::decorated_iterator<base_iterator, IteratorTag>
        decorated_iterator;

    std::vector<std::size_t> c(10007);
    //fill first half of array with even numbers and second half
    //with odd numbers
    std::iota(std::begin(c), std::end(c), 0);

    bool caught_exception = false;
    try{
        hpx::parallel::is_sorted_until(policy,
            decorated_iterator(
                std::begin(c),
                [](){ throw std::runtime_error("test"); }),
            decorated_iterator(
                std::end(c),
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
void test_sorted_until_async_exception(ExPolicy p, IteratorTag)
{
    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::decorated_iterator<base_iterator, IteratorTag>
        decorated_iterator;

    std::vector<std::size_t> c(10007);
    std::iota(std::begin(c), std::end(c), 0);

    bool caught_exception = false;
    try {
        hpx::future<decorated_iterator> f =
            hpx::parallel::is_sorted_until(p,
                decorated_iterator(
                    std::begin(c),
                    [](){ throw std::runtime_error("test"); }),
                decorated_iterator(
                    std::end(c),
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
void test_sorted_until_exception()
{
    using namespace hpx::parallel;
    //If the execution policy object is of type vector_execution_policy,
    //  std::terminate shall be called. Therefore we do not test exceptions
    //  with a vector execution policy
    test_sorted_until_exception(execution::seq, IteratorTag());
    test_sorted_until_exception(execution::par, IteratorTag());

    test_sorted_until_async_exception(execution::seq(execution::task),
        IteratorTag());
    test_sorted_until_async_exception(execution::par(execution::task),
        IteratorTag());

#if defined(HPX_HAVE_GENERIC_EXECUTION_POLICY)
    test_sorted_until_exception(execution_policy(execution::par), IteratorTag());
    test_sorted_until_exception(execution_policy(execution::seq(execution::task)),
        IteratorTag());
    test_sorted_until_exception(execution_policy(execution::par(execution::task)),
        IteratorTag());
#endif
}

void sorted_until_exception_test()
{
    test_sorted_until_exception<std::random_access_iterator_tag>();
    test_sorted_until_exception<std::forward_iterator_tag>();
}

////////////////////////////////////////////////////////////////////////////////
template <typename ExPolicy, typename IteratorTag>
void test_sorted_until_bad_alloc(ExPolicy policy, IteratorTag)
{
    static_assert(
        hpx::parallel::execution::is_execution_policy<ExPolicy>::value,
        "hpx::parallel::execution::is_execution_policy<ExPolicy>::value");

    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::decorated_iterator<base_iterator, IteratorTag>
        decorated_iterator;

    std::vector<std::size_t> c(10007);
    //fill first half of array with even numbers and second half
    //with odd numbers
    std::fill(std::begin(c), std::begin(c) + c.size()/2,
        2*(std::rand() % 100));
    std::fill(std::begin(c) + c.size()/2, std::end(c),
        2*(std::rand() % 100) + 1);

    bool caught_bad_alloc = false;
    try {
        hpx::parallel::is_sorted_until(policy,
            decorated_iterator(
                std::begin(c),
                [](){ throw std::bad_alloc(); }),
            decorated_iterator(
                std::end(c),
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
void test_sorted_until_async_bad_alloc(ExPolicy p, IteratorTag)
{
    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::decorated_iterator<base_iterator, IteratorTag>
        decorated_iterator;

    std::vector<std::size_t> c(10007);
    //fill first half of array with even numbers and second half
    //with odd numbers
    std::fill(std::begin(c), std::begin(c) + c.size()/2,
        2*(std::rand() % 100));
    std::fill(std::begin(c) + c.size()/2, std::end(c),
        2*(std::rand() % 100) + 1);

    bool caught_bad_alloc = false;
    try {
        hpx::future<decorated_iterator> f =
            hpx::parallel::is_sorted_until(p,
                decorated_iterator(
                    std::begin(c),
                    [](){ throw std::bad_alloc(); }),
                decorated_iterator(
                    std::end(c),
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
void test_sorted_until_bad_alloc()
{
    using namespace hpx::parallel;

    // If the execution policy object is of type vector_execution_policy,
    // std::terminate shall be called. therefore we do not test exceptions
    // with a vector execution policy
    test_sorted_until_bad_alloc(execution::par, IteratorTag());
    test_sorted_until_bad_alloc(execution::seq, IteratorTag());

    test_sorted_until_async_bad_alloc(execution::seq(execution::task),
        IteratorTag());
    test_sorted_until_async_bad_alloc(execution::par(execution::task),
        IteratorTag());

#if defined(HPX_HAVE_GENERIC_EXECUTION_POLICY)
    test_sorted_until_bad_alloc(execution_policy(execution::par), IteratorTag());
    test_sorted_until_bad_alloc(execution_policy(execution::seq), IteratorTag());
    test_sorted_until_bad_alloc(execution_policy(execution::seq(execution::task)),
        IteratorTag());
    test_sorted_until_bad_alloc(execution_policy(execution::par(execution::task)),
        IteratorTag());
#endif
}

void sorted_until_bad_alloc_test()
{
    test_sorted_until_bad_alloc<std::random_access_iterator_tag>();
    test_sorted_until_bad_alloc<std::forward_iterator_tag>();
}


int hpx_main(boost::program_options::variables_map& vm)
{
    sorted_until_test1();
    sorted_until_test2();
    sorted_until_test3();
    sorted_until_exception_test();
    sorted_until_bad_alloc_test();
    return hpx::finalize();
}

int main(int argc, char* argv[])
{
    using namespace boost::program_options;
    options_description desc_commandline(
        "Usage: " HPX_APPLICATION_STRING " [options]");

    std::vector<std::string> const cfg = {
        "hpx.os_threads=all"
    };

    HPX_TEST_EQ_MSG(hpx::init(desc_commandline, argc, argv, cfg), 0,
        "HPX main exited with non-zero status");

    return hpx::util::report_errors();
}
