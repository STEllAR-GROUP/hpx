//  Copyright (c) 2014 Grant Mercer
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>
#include <hpx/hpx.hpp>
#include <hpx/include/parallel_count.hpp>
#include <hpx/util/lightweight_test.hpp>

#include "test_utils.hpp"

////////////////////////////////////////////////////////////////////////////
template <typename ExPolicy, typename IteratorTag>
void test_count(ExPolicy const& policy, IteratorTag)
{
    BOOST_STATIC_ASSERT(hpx::parallel::is_execution_policy<ExPolicy>::value);

    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<std::size_t> c(10007);
    //assure rand() does not evalulate to zero
    std::iota(boost::begin(c), boost::end(c), std::rand()+1);

    std::size_t find_count = (std::rand() % 30) + 1;
    for (std::size_t i = 0; i != find_count && i != c.size(); ++i)
    {
        c[i] = 0;
    }

    boost::int64_t num_items = hpx::parallel::count(policy,
        iterator(boost::begin(c)), iterator(boost::end(c)), std::size_t(0));

    HPX_TEST_EQ(boost::int64_t(num_items), find_count);

}

template <typename IteratorTag>
void test_count(hpx::parallel::task_execution_policy, IteratorTag)
{
    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<std::size_t> c(10007);
    //assure rand() does not evaluate to zero
    std::iota(boost::begin(c), boost::end(c), std::rand()+1);

    std::size_t find_count = (std::rand() % 30) + 1;
    for (std::size_t i = 0; i != find_count && i != c.size(); ++i)
    {
        c[i] = 0;
    }

    hpx::future<boost::int64_t> f =
        hpx::parallel::count(hpx::parallel::task,
            iterator(boost::begin(c)), iterator(boost::end(c)),
            std::size_t(0));

    HPX_TEST_EQ(boost::int64_t(find_count), f.get());
}

template <typename IteratorTag>
void test_count()
{
    using namespace hpx::parallel;
    test_count(seq, IteratorTag());
    test_count(par, IteratorTag());
    test_count(par_vec, IteratorTag());
    test_count(task, IteratorTag());

    test_count(execution_policy(seq), IteratorTag());
    test_count(execution_policy(par), IteratorTag());
    test_count(execution_policy(par_vec), IteratorTag());
    test_count(execution_policy(task), IteratorTag());
}

void count_test()
{
    test_count<std::random_access_iterator_tag>();
    test_count<std::forward_iterator_tag>();
    test_count<std::input_iterator_tag>();
}


//////////////////////////////////////////////////////////////////////////////
template <typename ExPolicy, typename IteratorTag>
void test_count_exception(ExPolicy const& policy, IteratorTag)
{
    BOOST_STATIC_ASSERT(hpx::parallel::is_execution_policy<ExPolicy>::value);

    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::decorated_iterator<base_iterator, IteratorTag>
        decorated_iterator;
    std::vector<std::size_t> c(10007);
    std::iota(boost::begin(c), boost::end(c), std::rand());

    bool caught_exception = false;
    try {
        hpx::parallel::count(policy,
            decorated_iterator(
                boost::begin(c),
                [](){ throw std::runtime_error("test"); }),
            decorated_iterator(boost::end(c)),
            std::size_t(10));
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
void test_count_exception(hpx::parallel::task_execution_policy, IteratorTag)
{
    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::decorated_iterator<base_iterator, IteratorTag>
        decorated_iterator;

    std::vector<std::size_t> c(10007);
    std::fill(boost::begin(c), boost::end(c), 10);

    bool caught_exception = false;
    try {
        hpx::future<boost::int64_t> f =
            hpx::parallel::count(hpx::parallel::task,
                decorated_iterator(
                    boost::begin(c),
                    [](){ throw std::runtime_error("test"); }),
                decorated_iterator(boost::end(c)),
                std::size_t(10));
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
void test_count_exception()
{
    using namespace hpx::parallel;

    // If the execution policy object is of type parallel_vector_execution_policy,
    // std::terminate shall be called. Therefore we do not test exceptions
    // with a vector execution policy.
    test_count_exception(seq, IteratorTag());
    test_count_exception(par, IteratorTag());
    test_count_exception(task, IteratorTag());

    test_count_exception(execution_policy(seq), IteratorTag());
    test_count_exception(execution_policy(par), IteratorTag());
    test_count_exception(execution_policy(task), IteratorTag());
}

void count_exception_test()
{
    test_count_exception<std::random_access_iterator_tag>();
    test_count_exception<std::forward_iterator_tag>();
    test_count_exception<std::input_iterator_tag>();
}

//////////////////////////////////////////////////////////////////////////////
template <typename ExPolicy, typename IteratorTag>
void test_count_bad_alloc(ExPolicy const& policy, IteratorTag)
{
    BOOST_STATIC_ASSERT(hpx::parallel::is_execution_policy<ExPolicy>::value);

    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::decorated_iterator<base_iterator, IteratorTag>
        decorated_iterator;

    std::vector<std::size_t> c(10007);
    std::iota(boost::begin(c), boost::end(c), std::rand());

    bool caught_bad_alloc = false;
    try {
        hpx::parallel::count(policy,
            decorated_iterator(
                boost::begin(c),
                [](){ throw std::bad_alloc(); }),
            decorated_iterator(boost::end(c)),
            std::size_t(10));
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
void test_count_bad_alloc(hpx::parallel::task_execution_policy, IteratorTag)
{
    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::decorated_iterator<base_iterator, IteratorTag>
        decorated_iterator;

    std::vector<std::size_t> c(10007);
    std::iota(boost::begin(c), boost::end(c), std::rand());

    bool caught_bad_alloc = false;
    try {
        hpx::future<boost::int64_t> f =
            hpx::parallel::count(hpx::parallel::task,
                decorated_iterator(
                    boost::begin(c),
                    [](){ throw std::bad_alloc(); }),
                decorated_iterator(boost::end(c)),
                std::size_t(10));

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
void test_count_bad_alloc()
{
    using namespace hpx::parallel;

    // If the execution policy object is of type parallel_vector_execution_policy,
    // std::terminate shall be called. therefore we do not test exceptions
    // with a vector execution policy
    test_count_bad_alloc(seq, IteratorTag());
    test_count_bad_alloc(par, IteratorTag());
    test_count_bad_alloc(task, IteratorTag());

    test_count_bad_alloc(execution_policy(seq), IteratorTag());
    test_count_bad_alloc(execution_policy(par), IteratorTag());
    test_count_bad_alloc(execution_policy(task), IteratorTag());
}

void count_bad_alloc_test()
{
    test_count_bad_alloc<std::random_access_iterator_tag>();
    test_count_bad_alloc<std::forward_iterator_tag>();
    test_count_bad_alloc<std::input_iterator_tag>();
}

int hpx_main()
{
    count_test();
    count_exception_test();
    count_bad_alloc_test();
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
