//  Copyright (c) 2007-2014 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>
#include <hpx/hpx.hpp>
#include <hpx/include/parallel_rotate.hpp>
#include <hpx/util/lightweight_test.hpp>

#include "test_utils.hpp"

///////////////////////////////////////////////////////////////////////////////
template <typename ExPolicy, typename IteratorTag>
void test_rotate_copy(ExPolicy const& policy, IteratorTag)
{
    BOOST_STATIC_ASSERT(hpx::parallel::is_execution_policy<ExPolicy>::value);

    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<std::size_t> c(10007);
    std::vector<std::size_t> d1(c.size());
    std::vector<std::size_t> d2(c.size());

    std::iota(boost::begin(c), boost::end(c), std::rand());

    base_iterator mid = boost::begin(c);
    std::advance(mid, std::rand() % c.size());

    hpx::parallel::rotate_copy(policy,
        iterator(boost::begin(c)), iterator(mid), iterator(boost::end(c)),
        boost::begin(d1));

    std::rotate_copy(boost::begin(c), mid, boost::end(c), boost::begin(d2));

    std::size_t count = 0;
    HPX_TEST(std::equal(boost::begin(d1), boost::end(d1), boost::begin(d2),
        [&count](std::size_t v1, std::size_t v2) {
            HPX_TEST_EQ(v1, v2);
            ++count;
            return v1 == v2;
        }));
    HPX_TEST_EQ(count, d1.size());
}

template <typename IteratorTag>
void test_rotate_copy(hpx::parallel::task_execution_policy, IteratorTag)
{
    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<std::size_t> c(10007);
    std::vector<std::size_t> d1(c.size());
    std::vector<std::size_t> d2(c.size());

    std::iota(boost::begin(c), boost::end(c), std::rand());

    base_iterator mid = boost::begin(c);
    std::advance(mid, std::rand() % c.size());

    auto f =
        hpx::parallel::rotate_copy(hpx::parallel::task,
            iterator(boost::begin(c)), iterator(mid), iterator(boost::end(c)),
            boost::begin(d1));
    f.wait();

    std::rotate_copy(boost::begin(c), mid, boost::end(c), boost::begin(d2));

    std::size_t count = 0;
    HPX_TEST(std::equal(boost::begin(d1), boost::end(d1), boost::begin(d2),
        [&count](std::size_t v1, std::size_t v2) {
            HPX_TEST_EQ(v1, v2);
            ++count;
            return v1 == v2;
        }));
    HPX_TEST_EQ(count, d1.size());
}

template <typename IteratorTag>
void test_rotate_copy()
{
    using namespace hpx::parallel;
    test_rotate_copy(seq, IteratorTag());
    test_rotate_copy(par, IteratorTag());
    test_rotate_copy(par_vec, IteratorTag());
    test_rotate_copy(task, IteratorTag());

    test_rotate_copy(execution_policy(seq), IteratorTag());
    test_rotate_copy(execution_policy(par), IteratorTag());
    test_rotate_copy(execution_policy(par_vec), IteratorTag());
    test_rotate_copy(execution_policy(task), IteratorTag());
}

void rotate_copy_test()
{
    test_rotate_copy<std::random_access_iterator_tag>();
    test_rotate_copy<std::forward_iterator_tag>();
}

///////////////////////////////////////////////////////////////////////////////
template <typename ExPolicy, typename IteratorTag>
void test_rotate_copy_exception(ExPolicy const& policy, IteratorTag)
{
    BOOST_STATIC_ASSERT(hpx::parallel::is_execution_policy<ExPolicy>::value);

    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::decorated_iterator<base_iterator, IteratorTag>
        decorated_iterator;

    std::vector<std::size_t> c(10007);
    std::vector<std::size_t> d(c.size());
    std::iota(boost::begin(c), boost::end(c), std::rand());

    base_iterator mid = boost::begin(c);
    std::advance(mid, std::rand() % c.size());

    bool caught_exception = false;
    try {
        hpx::parallel::rotate_copy(policy,
            decorated_iterator(
                boost::begin(c),
                [](){ throw std::runtime_error("test"); }),
            decorated_iterator(mid),
            decorated_iterator(boost::end(c)),
            boost::begin(d));
        HPX_TEST(false);
    }
    catch (hpx::exception_list const& e) {
        caught_exception = true;
        test::test_num_exceptions<ExPolicy, IteratorTag>::call(policy, e);
    }
    catch (...) {
        HPX_TEST(false);
    }

    HPX_TEST(caught_exception);
}

template <typename IteratorTag>
void test_rotate_copy_exception(hpx::parallel::task_execution_policy, IteratorTag)
{
    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::decorated_iterator<base_iterator, IteratorTag>
        decorated_iterator;

    std::vector<std::size_t> c(10007);
    std::vector<std::size_t> d(c.size());
    std::iota(boost::begin(c), boost::end(c), std::rand());

    base_iterator mid = boost::begin(c);
    std::advance(mid, std::rand() % c.size());

    bool caught_exception = false;
    try {
        hpx::future<base_iterator> f =
            hpx::parallel::rotate_copy(hpx::parallel::task,
                decorated_iterator(
                    boost::begin(c),
                    [](){ throw std::runtime_error("test"); }),
                decorated_iterator(mid),
                decorated_iterator(boost::end(c)),
                boost::begin(d));
        f.get();

        HPX_TEST(false);
    }
    catch (hpx::exception_list const& e) {
        caught_exception = true;
        test::test_num_exceptions<
            hpx::parallel::task_execution_policy, IteratorTag
        >::call(hpx::parallel::task, e);
    }
    catch (...) {
        HPX_TEST(false);
    }

    HPX_TEST(caught_exception);
}

template <typename IteratorTag>
void test_rotate_copy_exception()
{
    using namespace hpx::parallel;

    // If the execution policy object is of type vector_execution_policy,
    // std::terminate shall be called. therefore we do not test exceptions
    // with a vector execution policy
    test_rotate_copy_exception(seq, IteratorTag());
    test_rotate_copy_exception(par, IteratorTag());
    test_rotate_copy_exception(task, IteratorTag());

    test_rotate_copy_exception(execution_policy(seq), IteratorTag());
    test_rotate_copy_exception(execution_policy(par), IteratorTag());
    test_rotate_copy_exception(execution_policy(task), IteratorTag());
}

void rotate_copy_exception_test()
{
    test_rotate_copy_exception<std::random_access_iterator_tag>();
    test_rotate_copy_exception<std::forward_iterator_tag>();
}

//////////////////////////////////////////////////////////////////////////////
template <typename ExPolicy, typename IteratorTag>
void test_rotate_copy_bad_alloc(ExPolicy const& policy, IteratorTag)
{
    BOOST_STATIC_ASSERT(hpx::parallel::is_execution_policy<ExPolicy>::value);

    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::decorated_iterator<base_iterator, IteratorTag>
        decorated_iterator;

    std::vector<std::size_t> c(10007);
    std::vector<std::size_t> d(c.size());
    std::iota(boost::begin(c), boost::end(c), std::rand());

    base_iterator mid = boost::begin(c);
    std::advance(mid, std::rand() % c.size());

    bool caught_bad_alloc = false;
    try {
        hpx::parallel::rotate_copy(policy,
            decorated_iterator(
                boost::begin(c),
                [](){ throw std::bad_alloc(); }),
            decorated_iterator(mid),
            decorated_iterator(boost::end(c)),
            boost::begin(d));
        HPX_TEST(false);
    }
    catch (std::bad_alloc const&) {
        caught_bad_alloc = true;
    }
    catch (...) {
        HPX_TEST(false);
    }

    HPX_TEST(caught_bad_alloc);
}

template <typename IteratorTag>
void test_rotate_copy_bad_alloc(hpx::parallel::task_execution_policy, IteratorTag)
{
    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::decorated_iterator<base_iterator, IteratorTag>
        decorated_iterator;

    std::vector<std::size_t> c(10007);
    std::vector<std::size_t> d(c.size());
    std::iota(boost::begin(c), boost::end(c), std::rand());

    base_iterator mid = boost::begin(c);
    std::advance(mid, std::rand() % c.size());

    bool caught_bad_alloc = false;
    try {
        hpx::future<base_iterator> f =
            hpx::parallel::rotate_copy(hpx::parallel::task,
                decorated_iterator(
                    boost::begin(c),
                    [](){ throw std::bad_alloc(); }),
                decorated_iterator(mid),
                decorated_iterator(boost::end(c)),
                boost::begin(d));

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
void test_rotate_copy_bad_alloc()
{
    using namespace hpx::parallel;
    //If the execution policy object is of type vector_execution_policy,
    //  std::terminate shall be called. therefore we do not test exceptions
    //  with a vector execution policy
    test_rotate_copy_bad_alloc(seq, IteratorTag());
    test_rotate_copy_bad_alloc(par, IteratorTag());
    test_rotate_copy_bad_alloc(task, IteratorTag());

    test_rotate_copy_bad_alloc(execution_policy(seq), IteratorTag());
    test_rotate_copy_bad_alloc(execution_policy(par), IteratorTag());
    test_rotate_copy_bad_alloc(execution_policy(task), IteratorTag());
}

void rotate_copy_bad_alloc_test()
{
    test_rotate_copy_bad_alloc<std::random_access_iterator_tag>();
    test_rotate_copy_bad_alloc<std::forward_iterator_tag>();
}

int hpx_main()
{
    rotate_copy_test();
    rotate_copy_exception_test();
    rotate_copy_bad_alloc_test();
    return hpx::finalize();
}

int main(int argc, char* argv[])
{
    std::vector<std::string> cfg;
    cfg.push_back("hpx.os_threads=" +
        boost::lexical_cast<std::string>(hpx::threads::hardware_concurrency()));

    HPX_TEST_EQ_MSG(hpx::init(argc, argv, cfg), 0,
        "HPX main exited with non-zero status");

    return hpx::util::report_errors();

}
