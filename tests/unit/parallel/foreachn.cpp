//  Copyright (c) 2014 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>
#include <hpx/hpx.hpp>
#include <hpx/include/parallel_for_each.hpp>
#include <hpx/util/lightweight_test.hpp>

#include "test_utils.hpp"

///////////////////////////////////////////////////////////////////////////////
template <typename ExPolicy, typename IteratorTag>
void test_for_each_n(ExPolicy const& policy, IteratorTag)
{
    BOOST_STATIC_ASSERT(hpx::parallel::is_execution_policy<ExPolicy>::value);

    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<std::size_t> c(10007);
    std::iota(boost::begin(c), boost::end(c), std::rand());

    iterator result = hpx::parallel::for_each_n(policy,
        iterator(boost::begin(c)), c.size(),
        [](std::size_t& v) {
            v = 42;
        });
    iterator end = iterator(boost::end(c));
    HPX_TEST(result == end);

    // verify values
    std::size_t count = 0;
    std::for_each(boost::begin(c), boost::end(c),
        [&count](std::size_t v) {
            HPX_TEST_EQ(v, std::size_t(42));
            ++count;
        });
    HPX_TEST_EQ(count, c.size());
}

template <typename IteratorTag>
void test_for_each_n(hpx::parallel::task_execution_policy, IteratorTag)
{
    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<std::size_t> c(10007);
    std::iota(boost::begin(c), boost::end(c), std::rand());

    hpx::future<iterator> f =
        hpx::parallel::for_each_n(hpx::parallel::task,
            iterator(boost::begin(c)), c.size(),
            [](std::size_t& v) {
                v = 42;
            });
    HPX_TEST(f.get() == iterator(boost::end(c)));

    // verify values
    std::size_t count = 0;
    std::for_each(boost::begin(c), boost::end(c),
        [&count](std::size_t v) {
            HPX_TEST_EQ(v, std::size_t(42));
            ++count;
        });
    HPX_TEST_EQ(count, c.size());
}

template <typename IteratorTag>
void test_for_each_n()
{
    using namespace hpx::parallel;

    test_for_each_n(seq, IteratorTag());
    test_for_each_n(par, IteratorTag());
    test_for_each_n(par_vec, IteratorTag());
    test_for_each_n(task, IteratorTag());

    test_for_each_n(execution_policy(seq), IteratorTag());
    test_for_each_n(execution_policy(par), IteratorTag());
    test_for_each_n(execution_policy(par_vec), IteratorTag());
    test_for_each_n(execution_policy(task), IteratorTag());
}

void for_each_n_test()
{
    test_for_each_n<std::random_access_iterator_tag>();
    test_for_each_n<std::forward_iterator_tag>();
    test_for_each_n<std::input_iterator_tag>();
}

///////////////////////////////////////////////////////////////////////////////
template <typename ExPolicy, typename IteratorTag>
void test_for_each_n_exception(ExPolicy const& policy, IteratorTag)
{
    BOOST_STATIC_ASSERT(hpx::parallel::is_execution_policy<ExPolicy>::value);

    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<std::size_t> c(10007);
    std::iota(boost::begin(c), boost::end(c), std::rand());

    bool caught_exception = false;
    try {
        hpx::parallel::for_each_n(policy,
            iterator(boost::begin(c)), c.size(),
            [](std::size_t& v) {
                throw std::runtime_error("test");
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

template <typename IteratorTag>
void test_for_each_n_exception(hpx::parallel::task_execution_policy, IteratorTag)
{
    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<std::size_t> c(10007);
    std::iota(boost::begin(c), boost::end(c), std::rand());

    bool caught_exception = false;
    try {
        hpx::future<iterator> f =
            hpx::parallel::for_each_n(hpx::parallel::task,
                iterator(boost::begin(c)), c.size(),
                [](std::size_t& v) {
                    throw std::runtime_error("test");
                });
        f.get();    // rethrow exception

        HPX_TEST(false);
    }
    catch(hpx::exception_list const& e) {
        caught_exception = true;
        test::test_num_exceptions<
            hpx::parallel::task_execution_policy, IteratorTag
        >::call(hpx::parallel::task, e);
    }
    catch(...) {
        HPX_TEST(false);
    }

    HPX_TEST(caught_exception);
}

template <typename IteratorTag>
void test_for_each_n_exception()
{
    using namespace hpx::parallel;
    //If the execution policy object is of type vector_execution_policy,
    //  std::terminate shall be called. therefore we do not test exceptions
    //  with a vector execution policy
    test_for_each_n_exception(seq, IteratorTag());
    test_for_each_n_exception(par, IteratorTag());
    test_for_each_n_exception(task, IteratorTag());

    test_for_each_n_exception(execution_policy(seq), IteratorTag());
    test_for_each_n_exception(execution_policy(par), IteratorTag());
    test_for_each_n_exception(execution_policy(task), IteratorTag());
}

void for_each_n_exception_test()
{
    test_for_each_n_exception<std::random_access_iterator_tag>();
    test_for_each_n_exception<std::forward_iterator_tag>();
    test_for_each_n_exception<std::input_iterator_tag>();
}

///////////////////////////////////////////////////////////////////////////////
template <typename ExPolicy, typename IteratorTag>
void test_for_each_n_bad_alloc(ExPolicy const& policy, IteratorTag)
{
    BOOST_STATIC_ASSERT(hpx::parallel::is_execution_policy<ExPolicy>::value);

    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<std::size_t> c(10007);
    std::iota(boost::begin(c), boost::end(c), std::rand());

    bool caught_bad_alloc = false;
    try {
        hpx::parallel::for_each_n(policy,
            iterator(boost::begin(c)), c.size(),
            [](std::size_t& v) {
                throw std::bad_alloc();
            });

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
void test_for_each_n_bad_alloc(hpx::parallel::task_execution_policy, IteratorTag)
{
    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<std::size_t> c(10007);
    std::iota(boost::begin(c), boost::end(c), std::rand());

    bool caught_bad_alloc = false;
    try {
        hpx::future<iterator> f =
            hpx::parallel::for_each_n(hpx::parallel::task,
                iterator(boost::begin(c)), c.size(),
                [](std::size_t& v) {
                    throw std::bad_alloc();
                });
        f.get();    // rethrow bad_alloc

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
void test_for_each_n_bad_alloc()
{
    using namespace hpx::parallel;
    //If the execution policy object is of type vector_execution_policy,
    //  std::terminate shall be called. therefore we do not test exceptions
    //  with a vector execution policy
    test_for_each_n_bad_alloc(seq, IteratorTag());
    test_for_each_n_bad_alloc(par, IteratorTag());
    test_for_each_n_bad_alloc(task, IteratorTag());

    test_for_each_n_bad_alloc(execution_policy(seq), IteratorTag());
    test_for_each_n_bad_alloc(execution_policy(par), IteratorTag());
    test_for_each_n_bad_alloc(execution_policy(task), IteratorTag());
}

void for_each_n_bad_alloc_test()
{
    test_for_each_n_bad_alloc<std::random_access_iterator_tag>();
    test_for_each_n_bad_alloc<std::forward_iterator_tag>();
    test_for_each_n_bad_alloc<std::input_iterator_tag>();
}

///////////////////////////////////////////////////////////////////////////////
int hpx_main()
{
    for_each_n_test();
    for_each_n_exception_test();
    for_each_n_bad_alloc_test();
    return hpx::finalize();
}

int main(int argc, char* argv[])
{
    // By default this test should run on all available cores
    std::vector<std::string> cfg;
    cfg.push_back("hpx.os_threads=" +
        boost::lexical_cast<std::string>(hpx::threads::hardware_concurrency()));

    // Initialize and run HPX
    HPX_TEST_EQ_MSG(hpx::init(argc, argv, cfg), 0,
        "HPX main exited with non-zero status");

    return hpx::util::report_errors();
}


