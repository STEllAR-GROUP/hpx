//  Copyright (c) 2014 Grant Mercer
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>
#include <hpx/hpx.hpp>
#include <hpx/include/parallel_find.hpp>
#include <hpx/util/lightweight_test.hpp>

#include "test_utils.hpp"

////////////////////////////////////////////////////////////////////////////
template <typename ExPolicy, typename IteratorTag>
void test_find_end1(ExPolicy const& policy, IteratorTag)
{
    BOOST_STATIC_ASSERT(hpx::parallel::is_execution_policy<ExPolicy>::value);

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

template <typename IteratorTag>
void test_find_end1(hpx::parallel::task_execution_policy, IteratorTag)
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
        hpx::parallel::find_end(hpx::parallel::task,
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
    test_find_end1(seq, IteratorTag());
    test_find_end1(par, IteratorTag());
    test_find_end1(par_vec, IteratorTag());
    test_find_end1(task, IteratorTag());


    test_find_end1(execution_policy(seq), IteratorTag());
    test_find_end1(execution_policy(par), IteratorTag());
    test_find_end1(execution_policy(par_vec), IteratorTag());
    test_find_end1(execution_policy(task), IteratorTag());

}

void find_end_test1()
{
    test_find_end1<std::random_access_iterator_tag>();
    test_find_end1<std::forward_iterator_tag>();
}

template <typename ExPolicy, typename IteratorTag>
void test_find_end2(ExPolicy const& policy, IteratorTag)
{
    BOOST_STATIC_ASSERT(hpx::parallel::is_execution_policy<ExPolicy>::value);

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

template <typename IteratorTag>
void test_find_end2(hpx::parallel::task_execution_policy, IteratorTag)
{
    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<std::size_t> c(10007);
    // fill vector with random values above 2
    std::fill(boost::begin(c), boost::end(c), (std::rand() % 100) + 3);
    // create subsequence at start and end
    c[0] = 1;
    c[1] = 2;
    c[c.size()-1] = 2;
    c[c.size()-2] = 1;

    std::size_t h[] = { 1, 2 };

    hpx::future<iterator> f =
        hpx::parallel::find_end(hpx::parallel::task,
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
    test_find_end2(seq, IteratorTag());
    test_find_end2(par, IteratorTag());
    test_find_end2(par_vec, IteratorTag());
    test_find_end2(task, IteratorTag());


    test_find_end2(execution_policy(seq), IteratorTag());
    test_find_end2(execution_policy(par), IteratorTag());
    test_find_end2(execution_policy(par_vec), IteratorTag());
    test_find_end2(execution_policy(task), IteratorTag());
}

void find_end_test2()
{
    test_find_end2<std::random_access_iterator_tag>();
    test_find_end2<std::forward_iterator_tag>();
}

template <typename ExPolicy, typename IteratorTag>
void test_find_end3(ExPolicy const& policy, IteratorTag)
{
    BOOST_STATIC_ASSERT(hpx::parallel::is_execution_policy<ExPolicy>::value);

    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<std::size_t> c(10007);
    // fill vector with random values above 2
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

template <typename IteratorTag>
void test_find_end3(hpx::parallel::task_execution_policy, IteratorTag)
{
    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<std::size_t> c(10007);
    // fill vector with random values above 6
    std::fill(boost::begin(c), boost::end(c), (std::rand() % 100) + 7);
    // create subsequence large enoush to always be split into multiple partitions
    std::iota(boost::begin(c), boost::begin(c) + c.size()/16+1, 1);
    std::size_t sub_size = c.size()/16 + 1;
    std::vector<std::size_t> h(sub_size);
    std::iota(boost::begin(h), boost::end(h), 1);

    // create only two partitions, splitting the desired sub sequence into
    // seperate partitions.
    hpx::future<iterator> f =
        hpx::parallel::find_end(hpx::parallel::task,
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
    test_find_end3(seq, IteratorTag());
    test_find_end3(par, IteratorTag());
    test_find_end3(par_vec, IteratorTag());
    test_find_end3(task, IteratorTag());


    test_find_end3(execution_policy(seq), IteratorTag());
    test_find_end3(execution_policy(par), IteratorTag());
    test_find_end3(execution_policy(par_vec), IteratorTag());
    test_find_end3(execution_policy(task), IteratorTag());
}

void find_end_test3()
{
    test_find_end3<std::random_access_iterator_tag>();
    test_find_end3<std::forward_iterator_tag>();
}

template <typename ExPolicy, typename IteratorTag>
void test_find_end4(ExPolicy const& policy, IteratorTag)
{
    BOOST_STATIC_ASSERT(hpx::parallel::is_execution_policy<ExPolicy>::value);

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

template <typename IteratorTag>
void test_find_end4(hpx::parallel::task_execution_policy, IteratorTag)
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
        hpx::parallel::find_end(hpx::parallel::task,
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
    test_find_end4(seq, IteratorTag());
    test_find_end4(par, IteratorTag());
    test_find_end4(par_vec, IteratorTag());
    test_find_end4(task, IteratorTag());


    test_find_end4(execution_policy(seq), IteratorTag());
    test_find_end4(execution_policy(par), IteratorTag());
    test_find_end4(execution_policy(par_vec), IteratorTag());
    test_find_end4(execution_policy(task), IteratorTag());
}

void find_end_test4()
{
    test_find_end4<std::random_access_iterator_tag>();
    test_find_end4<std::forward_iterator_tag>();
}


///////////////////////////////////////////////////////////////////////////////
template <typename ExPolicy, typename IteratorTag>
void test_find_end_exception(ExPolicy const& policy, IteratorTag)
{
    BOOST_STATIC_ASSERT(hpx::parallel::is_execution_policy<ExPolicy>::value);

    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::decorated_iterator<base_iterator, IteratorTag>
        decorated_iterator;
    std::vector<std::size_t> c(10007);
    std::iota(boost::begin(c), boost::end(c), std::rand() + 1);
    c[c.size()/2] = 0;

    std::size_t h[] = { 1, 2 };

    bool caught_exception = false;
    try {
        hpx::parallel::find_end(policy,
            decorated_iterator(
                boost::begin(c),
                [](){ throw std::runtime_error("test"); }),
            decorated_iterator(boost::end(c)),
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

template <typename IteratorTag>
void test_find_end_exception(hpx::parallel::task_execution_policy, IteratorTag)
{
    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::decorated_iterator<base_iterator, IteratorTag>
        decorated_iterator;

    std::vector<std::size_t> c(10007);
    std::iota(boost::begin(c), boost::end(c), std::rand() + 1);
    c[c.size()/2] = 0;

    std::size_t h[] = { 1, 2 };

    bool caught_exception = false;
    try {
        hpx::future<decorated_iterator> f =
            hpx::parallel::find_end(hpx::parallel::task,
                decorated_iterator(
                    boost::begin(c),
                    [](){ throw std::runtime_error("test"); }),
                decorated_iterator(boost::end(c)),
                boost::begin(h), boost::end(h));
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
void test_find_end_exception()
{
    using namespace hpx::parallel;
    //If the execution policy object is of type vector_execution_policy,
    //  std::terminate shall be called. therefore we do not test exceptions
    //  with a vector execution policy
    test_find_end_exception(seq, IteratorTag());
    test_find_end_exception(par, IteratorTag());
    test_find_end_exception(task, IteratorTag());

    test_find_end_exception(execution_policy(seq), IteratorTag());
    test_find_end_exception(execution_policy(par), IteratorTag());
    test_find_end_exception(execution_policy(task), IteratorTag());
}

void find_end_exception_test()
{
    test_find_end_exception<std::random_access_iterator_tag>();
    test_find_end_exception<std::forward_iterator_tag>();
}

//////////////////////////////////////////////////////////////////////////////
template <typename ExPolicy, typename IteratorTag>
void test_find_end_bad_alloc(ExPolicy const& policy, IteratorTag)
{
    BOOST_STATIC_ASSERT(hpx::parallel::is_execution_policy<ExPolicy>::value);

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
            decorated_iterator(boost::end(c)),
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

template <typename IteratorTag>
void test_find_end_bad_alloc(hpx::parallel::task_execution_policy, IteratorTag)
{
    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::decorated_iterator<base_iterator, IteratorTag>
        decorated_iterator;

    std::vector<std::size_t> c(10007);
    std::iota(boost::begin(c), boost::end(c), std::rand() + 1);
    c[c.size()/2] = 0;

    std::size_t h[] = { 1, 2 };

    bool caught_bad_alloc = false;
    try {
        hpx::future<decorated_iterator> f =
            hpx::parallel::find_end(hpx::parallel::task,
                decorated_iterator(
                    boost::begin(c),
                    [](){ throw std::bad_alloc(); }),
                decorated_iterator(boost::end(c)),
                boost::begin(h), boost::end(h));

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
void test_find_end_bad_alloc()
{
    using namespace hpx::parallel;
    //If the execution policy object is of type vector_execution_policy,
    //  std::terminate shall be called. therefore we do not test exceptions
    //  with a vector execution policy
    test_find_end_bad_alloc(seq, IteratorTag());
    test_find_end_bad_alloc(par, IteratorTag());
    test_find_end_bad_alloc(task, IteratorTag());

    test_find_end_bad_alloc(execution_policy(seq), IteratorTag());
    test_find_end_bad_alloc(execution_policy(par), IteratorTag());
    test_find_end_bad_alloc(execution_policy(task), IteratorTag());
}

void find_end_bad_alloc_test()
{

    test_find_end_bad_alloc<std::random_access_iterator_tag>();
    test_find_end_bad_alloc<std::forward_iterator_tag>();

}

int hpx_main()
{
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
    std::vector<std::string> cfg;
    cfg.push_back("hpx.os_threads=" +
        boost::lexical_cast<std::string>(hpx::threads::hardware_concurrency()));

    HPX_TEST_EQ_MSG(hpx::init(argc, argv, cfg), 0,
        "HPX main exited with non-zero status");

    return hpx::util::report_errors();
}
