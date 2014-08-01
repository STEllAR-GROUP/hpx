//  Copyright (c) 2014 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>
#include <hpx/hpx.hpp>
#include <hpx/include/parallel_reduce.hpp>
#include <hpx/util/lightweight_test.hpp>

#include "test_utils.hpp"

///////////////////////////////////////////////////////////////////////////////
template <typename ExPolicy, typename IteratorTag>
void test_reduce1(ExPolicy const& policy, IteratorTag)
{
    BOOST_STATIC_ASSERT(hpx::parallel::is_execution_policy<ExPolicy>::value);

    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<std::size_t> c(10007);
    std::iota(boost::begin(c), boost::end(c), std::rand());

    std::size_t const val(42);
    auto op =
        [val](std::size_t v1, std::size_t v2) {
            return v1 + v2 + val;
        };

    std::size_t r1 = hpx::parallel::reduce(policy,
        iterator(boost::begin(c)), iterator(boost::end(c)), val, op);

    // verify values
    std::size_t r2 = std::accumulate(boost::begin(c), boost::end(c), val, op);
    HPX_TEST_EQ(r1, r2);
}

template <typename IteratorTag>
void test_reduce1(hpx::parallel::task_execution_policy, IteratorTag)
{
    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<std::size_t> c(10007);
    std::iota(boost::begin(c), boost::end(c), std::rand());

    std::size_t const val(42);
    auto op =
        [val](std::size_t v1, std::size_t v2) {
            return v1 + v2 + val;
        };

    hpx::future<std::size_t> f =
        hpx::parallel::reduce(hpx::parallel::task,
            iterator(boost::begin(c)), iterator(boost::end(c)), val, op);
    f.wait();

    // verify values
    std::size_t r2 = std::accumulate(boost::begin(c), boost::end(c), val, op);
    HPX_TEST_EQ(f.get(), r2);
}

template <typename IteratorTag>
void test_reduce1()
{
    using namespace hpx::parallel;

    test_reduce1(seq, IteratorTag());
    test_reduce1(par, IteratorTag());
    test_reduce1(par_vec, IteratorTag());
    test_reduce1(task, IteratorTag());

    test_reduce1(execution_policy(seq), IteratorTag());
    test_reduce1(execution_policy(par), IteratorTag());
    test_reduce1(execution_policy(par_vec), IteratorTag());
    test_reduce1(execution_policy(task), IteratorTag());
}

void reduce_test1()
{
    test_reduce1<std::random_access_iterator_tag>();
    test_reduce1<std::forward_iterator_tag>();
    test_reduce1<std::input_iterator_tag>();
}

///////////////////////////////////////////////////////////////////////////////
template <typename ExPolicy, typename IteratorTag>
void test_reduce2(ExPolicy const& policy, IteratorTag)
{
    BOOST_STATIC_ASSERT(hpx::parallel::is_execution_policy<ExPolicy>::value);

    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<std::size_t> c(10007);
    std::iota(boost::begin(c), boost::end(c), std::rand());

    std::size_t const val(42);
    std::size_t r1 = hpx::parallel::reduce(policy,
        iterator(boost::begin(c)), iterator(boost::end(c)), val);

    // verify values
    std::size_t r2 = std::accumulate(boost::begin(c), boost::end(c), val);
    HPX_TEST_EQ(r1, r2);
}

template <typename IteratorTag>
void test_reduce2(hpx::parallel::task_execution_policy, IteratorTag)
{
    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<std::size_t> c(10007);
    std::iota(boost::begin(c), boost::end(c), std::rand());

    std::size_t const val(42);
    hpx::future<std::size_t> f =
        hpx::parallel::reduce(hpx::parallel::task,
            iterator(boost::begin(c)), iterator(boost::end(c)), val);
    f.wait();

    // verify values
    std::size_t r2 = std::accumulate(boost::begin(c), boost::end(c), val);
    HPX_TEST_EQ(f.get(), r2);
}

template <typename IteratorTag>
void test_reduce2()
{
    using namespace hpx::parallel;

    test_reduce2(seq, IteratorTag());
    test_reduce2(par, IteratorTag());
    test_reduce2(par_vec, IteratorTag());
    test_reduce2(task, IteratorTag());

    test_reduce2(execution_policy(seq), IteratorTag());
    test_reduce2(execution_policy(par), IteratorTag());
    test_reduce2(execution_policy(par_vec), IteratorTag());
    test_reduce2(execution_policy(task), IteratorTag());
}

void reduce_test2()
{
    test_reduce2<std::random_access_iterator_tag>();
    test_reduce2<std::forward_iterator_tag>();
    test_reduce2<std::input_iterator_tag>();
}

///////////////////////////////////////////////////////////////////////////////
template <typename ExPolicy, typename IteratorTag>
void test_reduce3(ExPolicy const& policy, IteratorTag)
{
    BOOST_STATIC_ASSERT(hpx::parallel::is_execution_policy<ExPolicy>::value);

    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<std::size_t> c(10007);
    std::iota(boost::begin(c), boost::end(c), std::rand());

    std::size_t r1 = hpx::parallel::reduce(policy,
        iterator(boost::begin(c)), iterator(boost::end(c)));

    // verify values
    std::size_t r2 = std::accumulate(boost::begin(c), boost::end(c), std::size_t(0));
    HPX_TEST_EQ(r1, r2);
}

template <typename IteratorTag>
void test_reduce3(hpx::parallel::task_execution_policy, IteratorTag)
{
    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<std::size_t> c(10007);
    std::iota(boost::begin(c), boost::end(c), std::rand());

    hpx::future<std::size_t> f =
        hpx::parallel::reduce(hpx::parallel::task,
            iterator(boost::begin(c)), iterator(boost::end(c)));
    f.wait();

    // verify values
    std::size_t r2 = std::accumulate(boost::begin(c), boost::end(c), std::size_t(0));
    HPX_TEST_EQ(f.get(), r2);
}

template <typename IteratorTag>
void test_reduce3()
{
    using namespace hpx::parallel;

    test_reduce3(seq, IteratorTag());
    test_reduce3(par, IteratorTag());
    test_reduce3(par_vec, IteratorTag());
    test_reduce3(task, IteratorTag());

    test_reduce3(execution_policy(seq), IteratorTag());
    test_reduce3(execution_policy(par), IteratorTag());
    test_reduce3(execution_policy(par_vec), IteratorTag());
    test_reduce3(execution_policy(task), IteratorTag());
}

void reduce_test3()
{
    test_reduce3<std::random_access_iterator_tag>();
    test_reduce3<std::forward_iterator_tag>();
    test_reduce3<std::input_iterator_tag>();
}

///////////////////////////////////////////////////////////////////////////////
template <typename ExPolicy, typename IteratorTag>
void test_reduce_exception(ExPolicy const& policy, IteratorTag)
{
    BOOST_STATIC_ASSERT(hpx::parallel::is_execution_policy<ExPolicy>::value);

    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<std::size_t> c(10007);
    std::iota(boost::begin(c), boost::end(c), std::rand());

    bool caught_exception = false;
    try {
        hpx::parallel::reduce(policy,
            iterator(boost::begin(c)), iterator(boost::end(c)),
            std::size_t(42),
            [](std::size_t v1, std::size_t v2) {
                throw std::runtime_error("test");
                return v1 + v2;
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
void test_reduce_exception(hpx::parallel::task_execution_policy, IteratorTag)
{
    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<std::size_t> c(10007);
    std::iota(boost::begin(c), boost::end(c), std::rand());

    bool caught_exception = false;
    try {
        hpx::future<void> f =
            hpx::parallel::reduce(hpx::parallel::task,
                iterator(boost::begin(c)), iterator(boost::end(c)),
                std::size_t(42),
                [](std::size_t v1, std::size_t v2) {
                    throw std::runtime_error("test");
                    return v1 + v2;
                });
        f.get();

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
void test_reduce_exception()
{
    using namespace hpx::parallel;
    //If the execution policy object is of type vector_execution_policy,
    //  std::terminate shall be called. therefore we do not test exceptions
    //  with a vector execution policy
    test_reduce_exception(seq, IteratorTag());
    test_reduce_exception(par, IteratorTag());
    test_reduce_exception(task, IteratorTag());

    test_reduce_exception(execution_policy(seq), IteratorTag());
    test_reduce_exception(execution_policy(par), IteratorTag());
    test_reduce_exception(execution_policy(task), IteratorTag());
}

void reduce_exception_test()
{
    test_reduce_exception<std::random_access_iterator_tag>();
    test_reduce_exception<std::forward_iterator_tag>();
    test_reduce_exception<std::input_iterator_tag>();
}

///////////////////////////////////////////////////////////////////////////////
template <typename ExPolicy, typename IteratorTag>
void test_reduce_bad_alloc(ExPolicy const& policy, IteratorTag)
{
    BOOST_STATIC_ASSERT(hpx::parallel::is_execution_policy<ExPolicy>::value);

    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<std::size_t> c(10007);
    std::iota(boost::begin(c), boost::end(c), std::rand());

    bool caught_exception = false;
    try {
        hpx::parallel::reduce(policy,
            iterator(boost::begin(c)), iterator(boost::end(c)),
            std::size_t(42),
            [](std::size_t v1, std::size_t v2) {
                throw std::bad_alloc();
                return v1 + v2;
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

template <typename IteratorTag>
void test_reduce_bad_alloc(hpx::parallel::task_execution_policy, IteratorTag)
{
    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<std::size_t> c(10007);
    std::iota(boost::begin(c), boost::end(c), std::rand());

    bool caught_exception = false;
    try {
        hpx::future<void> f =
            hpx::parallel::reduce(hpx::parallel::task,
                iterator(boost::begin(c)), iterator(boost::end(c)),
                std::size_t(42),
                [](std::size_t v1, std::size_t v2) {
                    throw std::bad_alloc();
                    return v1 + v2;
                });
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
}

template <typename IteratorTag>
void test_reduce_bad_alloc()
{
    using namespace hpx::parallel;
    //If the execution policy object is of type vector_execution_policy,
    //  std::terminate shall be called. therefore we do not test exceptions
    //  with a vector execution policy
    test_reduce_bad_alloc(seq, IteratorTag());
    test_reduce_bad_alloc(par, IteratorTag());
    test_reduce_bad_alloc(task, IteratorTag());

    test_reduce_bad_alloc(execution_policy(seq), IteratorTag());
    test_reduce_bad_alloc(execution_policy(par), IteratorTag());
    test_reduce_bad_alloc(execution_policy(task), IteratorTag());
}

void reduce_bad_alloc_test()
{
    test_reduce_bad_alloc<std::random_access_iterator_tag>();
    test_reduce_bad_alloc<std::forward_iterator_tag>();
    test_reduce_bad_alloc<std::input_iterator_tag>();
}

///////////////////////////////////////////////////////////////////////////////
int hpx_main()
{
    reduce_test1();
    reduce_test2();
    reduce_test3();

    reduce_exception_test();
    reduce_bad_alloc_test();
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
