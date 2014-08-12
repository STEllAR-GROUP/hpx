//  Copyright (c) 2014 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>
#include <hpx/hpx.hpp>
#include <hpx/include/parallel_transform.hpp>
#include <hpx/util/lightweight_test.hpp>

#include "test_utils.hpp"

///////////////////////////////////////////////////////////////////////////////
template <typename ExPolicy, typename IteratorTag>
void test_transform_binary(ExPolicy const& policy, IteratorTag)
{
    BOOST_STATIC_ASSERT(hpx::parallel::is_execution_policy<ExPolicy>::value);

    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<std::size_t> c1(10007);
    std::vector<std::size_t> c2(c1.size());
    std::vector<std::size_t> d1(c1.size()); //-V656
    std::iota(boost::begin(c1), boost::end(c1), std::rand());
    std::iota(boost::begin(c2), boost::end(c2), std::rand());

    auto add =
        [](std::size_t v1, std::size_t v2) {
            return v1 + v2;
        };

    hpx::parallel::transform(policy,
        iterator(boost::begin(c1)), iterator(boost::end(c1)),
        boost::begin(c2), boost::begin(d1), add);

    // verify values
    std::vector<std::size_t> d2(c1.size());
    std::transform(boost::begin(c1), boost::end(c1),
        boost::begin(c2), boost::begin(d2), add);

    std::size_t count = 0;
    HPX_TEST(std::equal(boost::begin(d1), boost::end(d1), boost::begin(d2),
        [&count](std::size_t v1, std::size_t v2) {
            HPX_TEST_EQ(v1, v2);
            ++count;
            return v1 == v2;
        }));
    HPX_TEST_EQ(count, d2.size());
}

template <typename IteratorTag>
void test_transform_binary(hpx::parallel::task_execution_policy, IteratorTag)
{
    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<std::size_t> c1(10007);
    std::vector<std::size_t> c2(c1.size());
    std::vector<std::size_t> d1(c1.size()); //-V656
    std::iota(boost::begin(c1), boost::end(c1), std::rand());
    std::iota(boost::begin(c2), boost::end(c2), std::rand());

    auto add =
        [](std::size_t v1, std::size_t v2) {
            return v1 + v2;
        };

    hpx::future<base_iterator> f =
        hpx::parallel::transform(hpx::parallel::task,
            iterator(boost::begin(c1)), iterator(boost::end(c1)),
            boost::begin(c2), boost::begin(d1), add);
    f.wait();

    // verify values
    std::vector<std::size_t> d2(c1.size());
    std::transform(boost::begin(c1), boost::end(c1),
        boost::begin(c2), boost::begin(d2), add);

    std::size_t count = 0;
    HPX_TEST(std::equal(boost::begin(d1), boost::end(d1), boost::begin(d2),
        [&count](std::size_t v1, std::size_t v2) {
            HPX_TEST_EQ(v1, v2);
            ++count;
            return v1 == v2;
        }));
    HPX_TEST_EQ(count, d2.size());
}

template <typename IteratorTag>
void test_transform_binary()
{
    using namespace hpx::parallel;

    test_transform_binary(seq, IteratorTag());
    test_transform_binary(par, IteratorTag());
    test_transform_binary(par_vec, IteratorTag());
    test_transform_binary(task, IteratorTag());

    test_transform_binary(execution_policy(seq), IteratorTag());
    test_transform_binary(execution_policy(par), IteratorTag());
    test_transform_binary(execution_policy(par_vec), IteratorTag());
    test_transform_binary(execution_policy(task), IteratorTag());
}

void transform_binary_test()
{
    test_transform_binary<std::random_access_iterator_tag>();
    test_transform_binary<std::forward_iterator_tag>();
    test_transform_binary<std::input_iterator_tag>();
}

///////////////////////////////////////////////////////////////////////////////
template <typename ExPolicy, typename IteratorTag>
void test_transform_binary_exception(ExPolicy const& policy, IteratorTag)
{
    BOOST_STATIC_ASSERT(hpx::parallel::is_execution_policy<ExPolicy>::value);

    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<std::size_t> c1(10007);
    std::vector<std::size_t> c2(c1.size());
    std::vector<std::size_t> d1(c1.size()); //-V656
    std::iota(boost::begin(c1), boost::end(c1), std::rand());
    std::iota(boost::begin(c2), boost::end(c2), std::rand());

    bool caught_exception = false;
    try {
        hpx::parallel::transform(policy,
            iterator(boost::begin(c1)), iterator(boost::end(c1)),
            boost::begin(c2), boost::begin(d1),
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
void test_transform_binary_exception(hpx::parallel::task_execution_policy, IteratorTag)
{
    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<std::size_t> c1(10007);
    std::vector<std::size_t> c2(c1.size());
    std::vector<std::size_t> d1(c1.size()); //-V656
    std::iota(boost::begin(c1), boost::end(c1), std::rand());
    std::iota(boost::begin(c2), boost::end(c2), std::rand());

    bool caught_exception = false;
    try {
        hpx::future<base_iterator> f =
            hpx::parallel::transform(hpx::parallel::task,
                iterator(boost::begin(c1)), iterator(boost::end(c1)),
                boost::begin(c2), boost::begin(d1),
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
void test_transform_binary_exception()
{
    using namespace hpx::parallel;
    //If the execution policy object is of type vector_execution_policy,
    //  std::terminate shall be called. therefore we do not test exceptions
    //  with a vector execution policy
    test_transform_binary_exception(seq, IteratorTag());
    test_transform_binary_exception(par, IteratorTag());
    test_transform_binary_exception(task, IteratorTag());

    test_transform_binary_exception(execution_policy(seq), IteratorTag());
    test_transform_binary_exception(execution_policy(par), IteratorTag());
    test_transform_binary_exception(execution_policy(task), IteratorTag());
}

void transform_binary_exception_test()
{
    test_transform_binary_exception<std::random_access_iterator_tag>();
    test_transform_binary_exception<std::forward_iterator_tag>();
    test_transform_binary_exception<std::input_iterator_tag>();
}

///////////////////////////////////////////////////////////////////////////////
template <typename ExPolicy, typename IteratorTag>
void test_transform_binary_bad_alloc(ExPolicy const& policy, IteratorTag)
{
    BOOST_STATIC_ASSERT(hpx::parallel::is_execution_policy<ExPolicy>::value);

    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<std::size_t> c1(10007);
    std::vector<std::size_t> c2(c1.size());
    std::vector<std::size_t> d1(c1.size()); //-V656
    std::iota(boost::begin(c1), boost::end(c1), std::rand());
    std::iota(boost::begin(c2), boost::end(c2), std::rand());

    bool caught_bad_alloc = false;
    try {
        hpx::parallel::transform(policy,
            iterator(boost::begin(c1)), iterator(boost::end(c1)),
            boost::begin(c2), boost::begin(d1),
            [](std::size_t v1, std::size_t v2) {
                throw std::bad_alloc();
                return v1 + v2;
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
void test_transform_binary_bad_alloc(hpx::parallel::task_execution_policy, IteratorTag)
{
    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<std::size_t> c1(10007);
    std::vector<std::size_t> c2(c1.size());
    std::vector<std::size_t> d1(c1.size()); //-V656
    std::iota(boost::begin(c1), boost::end(c1), std::rand());
    std::iota(boost::begin(c2), boost::end(c2), std::rand());

    bool caught_bad_alloc = false;
    try {
        hpx::future<base_iterator> f =
            hpx::parallel::transform(hpx::parallel::task,
                iterator(boost::begin(c1)), iterator(boost::end(c1)),
                boost::begin(c2), boost::begin(d1),
                [](std::size_t v1, std::size_t v2) {
                    throw std::bad_alloc();
                    return v1 + v2;
                });
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
void test_transform_binary_bad_alloc()
{
    using namespace hpx::parallel;
    //If the execution policy object is of type vector_execution_policy,
    //  std::terminate shall be called. therefore we do not test exceptions
    //  with a vector execution policy
    test_transform_binary_bad_alloc(seq, IteratorTag());
    test_transform_binary_bad_alloc(par, IteratorTag());
    test_transform_binary_bad_alloc(task, IteratorTag());

    test_transform_binary_bad_alloc(execution_policy(seq), IteratorTag());
    test_transform_binary_bad_alloc(execution_policy(par), IteratorTag());
    test_transform_binary_bad_alloc(execution_policy(task), IteratorTag());
}

void transform_binary_bad_alloc_test()
{
    test_transform_binary_bad_alloc<std::random_access_iterator_tag>();
    test_transform_binary_bad_alloc<std::forward_iterator_tag>();
    test_transform_binary_bad_alloc<std::input_iterator_tag>();
}

///////////////////////////////////////////////////////////////////////////////
int hpx_main()
{
    transform_binary_test();
    transform_binary_exception_test();
    transform_binary_bad_alloc_test();
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


