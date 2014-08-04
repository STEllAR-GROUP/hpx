//  Copyright (c) 2007-2014 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>
#include <hpx/hpx.hpp>
#include <hpx/include/parallel_reverse.hpp>
#include <hpx/util/lightweight_test.hpp>

#include "test_utils.hpp"

///////////////////////////////////////////////////////////////////////////////
template <typename ExPolicy, typename IteratorTag>
void test_reverse(ExPolicy const& policy, IteratorTag)
{
    BOOST_STATIC_ASSERT(hpx::parallel::is_execution_policy<ExPolicy>::value);

    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<std::size_t> c(10007);
    std::vector<std::size_t> d1;

    std::iota(boost::begin(c), boost::end(c), std::rand());
    std::copy(boost::begin(c), boost::end(c), std::back_inserter(d1));

    hpx::parallel::reverse(policy,
        iterator(boost::begin(c)), iterator(boost::end(c)));

    std::reverse(boost::begin(d1), boost::end(d1));

    std::size_t count = 0;
    HPX_TEST(std::equal(boost::begin(c), boost::end(c), boost::begin(d1),
        [&count](std::size_t v1, std::size_t v2) {
            HPX_TEST_EQ(v1, v2);
            ++count;
            return v1 == v2;
        }));
    HPX_TEST_EQ(count, d1.size());
}

template <typename IteratorTag>
void test_reverse(hpx::parallel::task_execution_policy, IteratorTag)
{
    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<std::size_t> c(10007);
    std::vector<std::size_t> d1;

    std::iota(boost::begin(c), boost::end(c), std::rand());
    std::copy(boost::begin(c), boost::end(c), std::back_inserter(d1));

    auto f =
        hpx::parallel::reverse(hpx::parallel::task,
            iterator(boost::begin(c)), iterator(boost::end(c)));
    f.wait();

    std::reverse(boost::begin(d1), boost::end(d1));

    std::size_t count = 0;
    HPX_TEST(std::equal(boost::begin(c), boost::end(c), boost::begin(d1),
        [&count](std::size_t v1, std::size_t v2) {
            HPX_TEST_EQ(v1, v2);
            ++count;
            return v1 == v2;
        }));
    HPX_TEST_EQ(count, d1.size());
}

template <typename IteratorTag>
void test_reverse()
{
    using namespace hpx::parallel;
    test_reverse(seq, IteratorTag());
    test_reverse(par, IteratorTag());
    test_reverse(par_vec, IteratorTag());
    test_reverse(task, IteratorTag());

    test_reverse(execution_policy(seq), IteratorTag());
    test_reverse(execution_policy(par), IteratorTag());
    test_reverse(execution_policy(par_vec), IteratorTag());
    test_reverse(execution_policy(task), IteratorTag());
}

void reverse_test()
{
    test_reverse<std::random_access_iterator_tag>();
    test_reverse<std::bidirectional_iterator_tag>();
}

///////////////////////////////////////////////////////////////////////////////
template <typename ExPolicy, typename IteratorTag>
void test_reverse_exception(ExPolicy const& policy, IteratorTag)
{
    BOOST_STATIC_ASSERT(hpx::parallel::is_execution_policy<ExPolicy>::value);

    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::decorated_iterator<base_iterator, IteratorTag>
        decorated_iterator;

    std::vector<std::size_t> c(10007);
    std::iota(boost::begin(c), boost::end(c), std::rand());

    bool caught_exception = false;
    try {
        hpx::parallel::reverse(policy,
            decorated_iterator(boost::begin(c)),
            decorated_iterator(
                boost::end(c),
                [](){ throw std::runtime_error("test"); }
            ));
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
void test_reverse_exception(hpx::parallel::task_execution_policy, IteratorTag)
{
    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::decorated_iterator<base_iterator, IteratorTag>
        decorated_iterator;

    std::vector<std::size_t> c(10007);
    std::iota(boost::begin(c), boost::end(c), std::rand());

    bool caught_exception = false;
    try {
        hpx::future<void> f =
            hpx::parallel::reverse(hpx::parallel::task,
                decorated_iterator(boost::begin(c)),
                decorated_iterator(
                    boost::end(c),
                    [](){ throw std::runtime_error("test"); }
                ));
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
void test_reverse_exception()
{
    using namespace hpx::parallel;

    // If the execution policy object is of type vector_execution_policy,
    // std::terminate shall be called. therefore we do not test exceptions
    // with a vector execution policy
    test_reverse_exception(seq, IteratorTag());
    test_reverse_exception(par, IteratorTag());
    test_reverse_exception(task, IteratorTag());

    test_reverse_exception(execution_policy(seq), IteratorTag());
    test_reverse_exception(execution_policy(par), IteratorTag());
    test_reverse_exception(execution_policy(task), IteratorTag());
}

void reverse_exception_test()
{
    test_reverse_exception<std::random_access_iterator_tag>();
    test_reverse_exception<std::bidirectional_iterator_tag>();
}

//////////////////////////////////////////////////////////////////////////////
template <typename ExPolicy, typename IteratorTag>
void test_reverse_bad_alloc(ExPolicy const& policy, IteratorTag)
{
    BOOST_STATIC_ASSERT(hpx::parallel::is_execution_policy<ExPolicy>::value);

    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::decorated_iterator<base_iterator, IteratorTag>
        decorated_iterator;

    std::vector<std::size_t> c(10007);
    std::iota(boost::begin(c), boost::end(c), std::rand());

    bool caught_bad_alloc = false;
    try {
        hpx::parallel::reverse(policy,
            decorated_iterator(boost::begin(c)),
            decorated_iterator(
                boost::end(c),
                [](){ throw std::bad_alloc(); }
            ));
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
void test_reverse_bad_alloc(hpx::parallel::task_execution_policy, IteratorTag)
{
    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::decorated_iterator<base_iterator, IteratorTag>
        decorated_iterator;

    std::vector<std::size_t> c(10007);
    std::iota(boost::begin(c), boost::end(c), std::rand());

    bool caught_bad_alloc = false;
    try {
        hpx::future<void> f =
            hpx::parallel::reverse(hpx::parallel::task,
                decorated_iterator(boost::begin(c)),
                decorated_iterator(
                    boost::end(c),
                    [](){ throw std::bad_alloc(); }
                ));

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
void test_reverse_bad_alloc()
{
    using namespace hpx::parallel;
    
    // If the execution policy object is of type vector_execution_policy,
    // std::terminate shall be called. therefore we do not test exceptions
    // with a vector execution policy
    test_reverse_bad_alloc(seq, IteratorTag());
    test_reverse_bad_alloc(par, IteratorTag());
    test_reverse_bad_alloc(task, IteratorTag());

    test_reverse_bad_alloc(execution_policy(seq), IteratorTag());
    test_reverse_bad_alloc(execution_policy(par), IteratorTag());
    test_reverse_bad_alloc(execution_policy(task), IteratorTag());
}

void reverse_bad_alloc_test()
{
    test_reverse_bad_alloc<std::random_access_iterator_tag>();
    test_reverse_bad_alloc<std::bidirectional_iterator_tag>();
}

int hpx_main()
{
    reverse_test();
    reverse_exception_test();
    reverse_bad_alloc_test();
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
