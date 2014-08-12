//  Copyright (c) 2014 Grant Mercer
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>
#include <hpx/hpx.hpp>
#include <hpx/include/parallel_swap_ranges.hpp>
#include <hpx/util/lightweight_test.hpp>

#include "test_utils.hpp"

///////////////////////////////////////////////////////////////////////////////
template <typename ExPolicy, typename IteratorTag>
void test_swap_ranges(ExPolicy const& policy, IteratorTag)
{
    BOOST_STATIC_ASSERT(hpx::parallel::is_execution_policy<ExPolicy>::value);

    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<std::size_t> c(10007);
    std::vector<std::size_t> d(c.size());
    std::iota(boost::begin(c), boost::end(c), std::rand());
    std::fill(boost::begin(d), boost::end(d), std::rand());

    hpx::parallel::swap_ranges(policy,
        iterator(boost::begin(c)), iterator(boost::end(c)), boost::begin(d));

    //equal begins at one, therefore counter is started at 1
    std::size_t count = 1;
    HPX_TEST(std::equal(boost::begin(c) + 1, boost::end(c), boost::begin(c),
        [&count](std::size_t v1, std::size_t v2) {
            HPX_TEST_EQ(v1, v2);
            ++count;
            return v1 == v2;
        }));
    HPX_TEST_EQ(count, c.size());

    count = 1;
    HPX_TEST(std::equal(boost::begin(d) + 1, boost::end(d), boost::begin(d),
        [&count](std::size_t v1, std::size_t v2) {
            HPX_TEST_NEQ(v1, v2);
            ++count;
            return !(v1 == v2);
        }));
    HPX_TEST_EQ(count, d.size());
}

template <typename IteratorTag>
void test_swap_ranges(hpx::parallel::task_execution_policy, IteratorTag)
{
    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<std::size_t> c(10007);
    std::vector<std::size_t> d(c.size());
    std::iota(boost::begin(c), boost::end(c), std::rand());
    std::fill(boost::begin(d), boost::end(d), std::rand());

    hpx::future<base_iterator> f =
        hpx::parallel::swap_ranges(hpx::parallel::task,
            iterator(boost::begin(c)), iterator(boost::end(c)), boost::begin(d));

    f.wait();

    std::size_t count = 1;
    HPX_TEST(std::equal(boost::begin(c) + 1, boost::end(c), boost::begin(c),
        [&count](std::size_t v1, std::size_t v2){
            HPX_TEST_EQ(v1, v2);
            ++count;
            return v1 == v2;
        }));

    count = 1;
    HPX_TEST(std::equal(boost::begin(d) + 1, boost::end(d), boost::begin(d),
        [&count](std::size_t v1, std::size_t v2){
            HPX_TEST_NEQ(v1, v2);
            ++count;
            return !(v1 == v2);
        }));
    HPX_TEST_EQ(count, d.size());
}

template <typename IteratorTag>
void test_swap_ranges()
{
    using namespace hpx::parallel;
    test_swap_ranges(seq, IteratorTag());
    test_swap_ranges(par, IteratorTag());
    test_swap_ranges(par_vec, IteratorTag());
    test_swap_ranges(task, IteratorTag());

    test_swap_ranges(execution_policy(seq), IteratorTag());
    test_swap_ranges(execution_policy(par), IteratorTag());
    test_swap_ranges(execution_policy(par_vec), IteratorTag());
    test_swap_ranges(execution_policy(task), IteratorTag());
}

void swap_ranges_test()
{
    test_swap_ranges<std::random_access_iterator_tag>();
    test_swap_ranges<std::forward_iterator_tag>();
}

///////////////////////////////////////////////////////////////////////////////
template <typename ExPolicy, typename IteratorTag>
void test_swap_ranges_exception(ExPolicy const& policy, IteratorTag)
{
    BOOST_STATIC_ASSERT(hpx::parallel::is_execution_policy<ExPolicy>::value);

    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::decorated_iterator<base_iterator, IteratorTag>
        decorated_iterator;

    std::vector<std::size_t> c(10007);
    std::vector<std::size_t> d(c.size());
    std::iota(boost::begin(c), boost::end(c), std::rand());
    std::fill(boost::begin(d), boost::end(d), std::rand());

    bool caught_exception = false;
    try {
        hpx::parallel::swap_ranges(policy,
            decorated_iterator(
                boost::begin(c),
                [](){ throw std::runtime_error("test"); }),
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
void test_swap_ranges_exception(hpx::parallel::task_execution_policy, IteratorTag)
{
    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::decorated_iterator<base_iterator, IteratorTag>
        decorated_iterator;

    std::vector<std::size_t> c(10007);
    std::vector<std::size_t> d(c.size());
    std::iota(boost::begin(c), boost::end(c), std::rand());
    std::fill(boost::begin(d), boost::end(d), std::rand());

    bool caught_exception = false;
    try {
        hpx::future<base_iterator> f =
            hpx::parallel::swap_ranges(hpx::parallel::task,
                decorated_iterator(
                    boost::begin(c),
                    [](){ throw std::runtime_error("test"); }),
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
void test_swap_ranges_exception()
{
    using namespace hpx::parallel;
    //If the execution policy object is of type vector_execution_policy,
    //  std::terminate shall be called. therefore we do not test exceptions
    //  with a vector execution policy
    test_swap_ranges_exception(seq, IteratorTag());
    test_swap_ranges_exception(par, IteratorTag());
    test_swap_ranges_exception(task, IteratorTag());

    test_swap_ranges_exception(execution_policy(seq), IteratorTag());
    test_swap_ranges_exception(execution_policy(par), IteratorTag());
    test_swap_ranges_exception(execution_policy(task), IteratorTag());
}

void swap_ranges_exception_test()
{
    test_swap_ranges_exception<std::random_access_iterator_tag>();
    test_swap_ranges_exception<std::forward_iterator_tag>();
}

//////////////////////////////////////////////////////////////////////////////
template <typename ExPolicy, typename IteratorTag>
void test_swap_ranges_bad_alloc(ExPolicy const& policy, IteratorTag)
{
    BOOST_STATIC_ASSERT(hpx::parallel::is_execution_policy<ExPolicy>::value);

    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::decorated_iterator<base_iterator, IteratorTag>
        decorated_iterator;

    std::vector<std::size_t> c(100007);
    std::vector<std::size_t> d(c.size());
    std::iota(boost::begin(c), boost::end(c), std::rand());
    std::fill(boost::begin(d), boost::end(d), std::rand());

    bool caught_bad_alloc = false;
    try {
        hpx::parallel::swap_ranges(policy,
            decorated_iterator(
                boost::begin(c),
                [](){ throw std::bad_alloc(); }),
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
void test_swap_ranges_bad_alloc(hpx::parallel::task_execution_policy, IteratorTag)
{
    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::decorated_iterator<base_iterator, IteratorTag>
        decorated_iterator;

    std::vector<std::size_t> c(10007);
    std::vector<std::size_t> d(c.size());
    std::iota(boost::begin(c), boost::end(c), std::rand());
    std::fill(boost::begin(d), boost::end(d), std::rand());

    bool caught_bad_alloc = false;
    try {
        hpx::future<base_iterator> f =
            hpx::parallel::swap_ranges(hpx::parallel::task,
                decorated_iterator(
                    boost::begin(c),
                    [](){ throw std::bad_alloc(); }),
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
void test_swap_ranges_bad_alloc()
{
    using namespace hpx::parallel;
    //If the execution policy object is of type vector_execution_policy,
    //  std::terminate shall be called. therefore we do not test exceptions
    //  with a vector execution policy
    test_swap_ranges_bad_alloc(seq, IteratorTag());
    test_swap_ranges_bad_alloc(par, IteratorTag());
    test_swap_ranges_bad_alloc(task, IteratorTag());

    test_swap_ranges_bad_alloc(execution_policy(seq), IteratorTag());
    test_swap_ranges_bad_alloc(execution_policy(par), IteratorTag());
    test_swap_ranges_bad_alloc(execution_policy(task), IteratorTag());
}

void swap_ranges_bad_alloc_test()
{
    test_swap_ranges_bad_alloc<std::random_access_iterator_tag>();
    test_swap_ranges_bad_alloc<std::forward_iterator_tag>();
}
int hpx_main()
{
    swap_ranges_test();
    swap_ranges_exception_test();
    swap_ranges_bad_alloc_test();
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
