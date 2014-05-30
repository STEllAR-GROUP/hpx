//  Copyright (c) 2014 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>
#include <hpx/hpx.hpp>
#include <hpx/include/algorithm.hpp>
#include <hpx/util/lightweight_test.hpp>

#include "test_iterator.hpp"

///////////////////////////////////////////////////////////////////////////////
template <typename ExPolicy, typename IteratorTag>
void test_for_each(ExPolicy const& policy, IteratorTag)
{
    BOOST_STATIC_ASSERT(hpx::parallel::is_execution_policy<ExPolicy>::value);

    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<std::size_t> c(10000);
    std::iota(boost::begin(c), boost::end(c), std::rand());

    hpx::parallel::for_each(policy,
        iterator(boost::begin(c)), iterator(boost::end(c)),
        [](std::size_t& v) {
            v = 42;
        });

    // verify values
    std::size_t count = 0;
    std::for_each(boost::begin(c), boost::end(c),
        [](std::size_t v) {
            HPX_TEST_EQ(v, std::size_t(42));
            ++count;
        });
    HPX_TEST_EQ(count, c.size());
}

template <typename IteratorTag>
void test_for_each(hpx::parallel::task_execution_policy, IteratorTag)
{
    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<std::size_t> c(10000);
    std::iota(boost::begin(c), boost::end(c), std::rand());

    hpx::future<void> f =
        hpx::parallel::for_each(hpx::parallel::task,
            iterator(boost::begin(c)), iterator(boost::end(c)),
            [](std::size_t& v) {
                v = 42;
            });
    f.wait();

    // verify values
    std::size_t count = 0;
    std::for_each(boost::begin(c), boost::end(c),
        [](std::size_t v) {
            HPX_TEST_EQ(v, std::size_t(42));
            ++count;
        });
    HPX_TEST_EQ(count, c.size());
}

template <typename IteratorTag>
void test_for_each()
{
    using namespace hpx::parallel;

    test_for_each(seq, IteratorTag());
    test_for_each(par, IteratorTag());
    test_for_each(vec, IteratorTag());
    test_for_each(task, IteratorTag());

    test_for_each(execution_policy(seq), IteratorTag());
    test_for_each(execution_policy(par), IteratorTag());
    test_for_each(execution_policy(vec), IteratorTag());
    test_for_each(execution_policy(task), IteratorTag());
}

void for_each_test()
{
    test_for_each<std::random_access_iterator_tag>();
    test_for_each<std::forward_iterator_tag>();
    test_for_each<std::input_iterator_tag>();
}

///////////////////////////////////////////////////////////////////////////////
template <typename ExPolicy, typename IteratorTag>
void test_for_each_exception(ExPolicy const& policy, IteratorTag)
{
    BOOST_STATIC_ASSERT(hpx::parallel::is_execution_policy<ExPolicy>::value);

    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<std::size_t> c(10000);
    std::iota(boost::begin(c), boost::end(c), std::rand());

    bool caught_exception = false;
    try {
        hpx::parallel::for_each(policy,
            iterator(boost::begin(c)), iterator(boost::end(c)),
            [](std::size_t& v) {
                throw std::runtime_error("test");
            });

        HPX_TEST(false);
    }
    catch(...) {
        caught_exception = true;
        boost::exception_ptr e = boost::current_exception();
    }

    HPX_TEST(caught_exception);
}

template <typename IteratorTag>
void test_for_each_exception(hpx::parallel::task_execution_policy, IteratorTag)
{
    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<std::size_t> c(10000);
    std::iota(boost::begin(c), boost::end(c), std::rand());

    bool caught_exception = false;
    try {
        hpx::future<void> f =
            hpx::parallel::for_each(hpx::parallel::task,
                iterator(boost::begin(c)), iterator(boost::end(c)),
                [](std::size_t& v) {
                    throw std::runtime_error("test");
                });
        f.get();

        HPX_TEST(false);
    }
    catch(...) {
        caught_exception = true;
        boost::exception_ptr e = boost::current_exception();
    }

    HPX_TEST(caught_exception);
}

template <typename IteratorTag>
void test_for_each_exception()
{
    using namespace hpx::parallel;

    test_for_each_exception(seq, IteratorTag());
    test_for_each_exception(par, IteratorTag());
    test_for_each_exception(vec, IteratorTag());
    test_for_each_exception(task, IteratorTag());

    test_for_each_exception(execution_policy(seq), IteratorTag());
    test_for_each_exception(execution_policy(par), IteratorTag());
    test_for_each_exception(execution_policy(vec), IteratorTag());
    test_for_each_exception(execution_policy(task), IteratorTag());
}

void for_each_exception_test()
{
    test_for_each_exception<std::random_access_iterator_tag>();
    test_for_each_exception<std::forward_iterator_tag>();
    test_for_each_exception<std::input_iterator_tag>();
}

///////////////////////////////////////////////////////////////////////////////
int hpx_main()
{
    for_each_test();
    for_each_exception_test();
    return hpx::finalize();
}

int main(int argc, char* argv[])
{
    // Initialize and run HPX
    HPX_TEST_EQ_MSG(hpx::init(argc, argv), 0,
        "HPX main exited with non-zero status");

    return hpx::util::report_errors();
}


