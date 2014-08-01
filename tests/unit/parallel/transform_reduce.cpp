//  Copyright (c) 2014 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>
#include <hpx/hpx.hpp>
#include <hpx/include/parallel_transform_reduce.hpp>
#include <hpx/util/lightweight_test.hpp>

#include "test_utils.hpp"
///////////////////////////////////////////////////////////////////////////////
template <typename ExPolicy, typename IteratorTag>
void test_transform_reduce(ExPolicy const& policy, IteratorTag)
{
    BOOST_STATIC_ASSERT(hpx::parallel::is_execution_policy<ExPolicy>::value);

    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<std::size_t> c(10007);
    std::iota(boost::begin(c), boost::end(c), std::rand());

    typedef hpx::util::tuple<std::size_t, std::size_t> result_type;

    using hpx::util::make_tuple;
    using hpx::util::get;

    auto reduce_op =
        [](result_type v1, result_type v2) -> result_type
        {
            return make_tuple(get<0>(v1)*get<0>(v2), get<1>(v1)*get<1>(v2));
        };

    auto convert_op =
        [](std::size_t val) -> result_type
        {
            return make_tuple(val, val);
        };

    result_type const init = make_tuple(std::size_t(1), std::size_t(1));

    result_type r1 =
        hpx::parallel::transform_reduce(policy,
            iterator(boost::begin(c)), iterator(boost::end(c)),
            init, reduce_op, convert_op);

    // verify values
    result_type r2 =
        std::accumulate(
            boost::begin(c), boost::end(c), init,
            [&reduce_op, &convert_op](result_type res, std::size_t val)
            {
                return reduce_op(res, convert_op(val));
            });

    HPX_TEST_EQ(get<0>(r1), get<0>(r2));
    HPX_TEST_EQ(get<1>(r1), get<1>(r2));
}

template <typename IteratorTag>
void test_transform_reduce(hpx::parallel::task_execution_policy, IteratorTag)
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
        hpx::parallel::transform_reduce(hpx::parallel::task,
        iterator(boost::begin(c)), iterator(boost::end(c)), val, op, [](std::size_t v){return v;});
    f.wait();

    // verify values
    std::size_t r2 = std::accumulate(boost::begin(c), boost::end(c), val, op);
    HPX_TEST_EQ(f.get(), r2);
}

template <typename IteratorTag>
void test_transform_reduce()
{
    using namespace hpx::parallel;

    test_transform_reduce(seq, IteratorTag());
    test_transform_reduce(par, IteratorTag());
    test_transform_reduce(par_vec, IteratorTag());
    test_transform_reduce(task, IteratorTag());

    test_transform_reduce(execution_policy(seq), IteratorTag());
    test_transform_reduce(execution_policy(par), IteratorTag());
    test_transform_reduce(execution_policy(par_vec), IteratorTag());
    test_transform_reduce(execution_policy(task), IteratorTag());
}

void transform_reduce_test()
{
    test_transform_reduce<std::random_access_iterator_tag>();
    test_transform_reduce<std::forward_iterator_tag>();
    test_transform_reduce<std::input_iterator_tag>();
}

///////////////////////////////////////////////////////////////////////////////
template <typename ExPolicy, typename IteratorTag>
void test_transform_reduce_exception(ExPolicy const& policy, IteratorTag)
{
    BOOST_STATIC_ASSERT(hpx::parallel::is_execution_policy<ExPolicy>::value);

    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<std::size_t> c(10007);
    std::iota(boost::begin(c), boost::end(c), std::rand());

    bool caught_exception = false;
    try {
        hpx::parallel::transform_reduce(policy,
            iterator(boost::begin(c)), iterator(boost::end(c)),
            std::size_t(42),
            [](std::size_t v1, std::size_t v2) {
                throw std::runtime_error("test");
                return v1 + v2;
            },
            [](std::size_t v){return v;}
        );

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
void test_transform_reduce_exception(hpx::parallel::task_execution_policy, IteratorTag)
{
    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<std::size_t> c(10007);
    std::iota(boost::begin(c), boost::end(c), std::rand());

    bool caught_exception = false;
    try {
        hpx::future<void> f =
            hpx::parallel::transform_reduce(hpx::parallel::task,
                iterator(boost::begin(c)), iterator(boost::end(c)),
                std::size_t(42),
                [](std::size_t v1, std::size_t v2) {
                    throw std::runtime_error("test");
                    return v1 + v2;
                },
                [](std::size_t v){return v;}
            );
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
void test_transform_reduce_exception()
{
    using namespace hpx::parallel;
    //If the execution policy object is of type vector_execution_policy,
    //  std::terminate shall be called. therefore we do not test exceptions
    //  with a vector execution policy
    test_transform_reduce_exception(seq, IteratorTag());
    test_transform_reduce_exception(par, IteratorTag());
    test_transform_reduce_exception(task, IteratorTag());

    test_transform_reduce_exception(execution_policy(seq), IteratorTag());
    test_transform_reduce_exception(execution_policy(par), IteratorTag());
    test_transform_reduce_exception(execution_policy(task), IteratorTag());
}

void transform_reduce_exception_test()
{
    test_transform_reduce_exception<std::random_access_iterator_tag>();
    test_transform_reduce_exception<std::forward_iterator_tag>();
    test_transform_reduce_exception<std::input_iterator_tag>();
}

///////////////////////////////////////////////////////////////////////////////
template <typename ExPolicy, typename IteratorTag>
void test_transform_reduce_bad_alloc(ExPolicy const& policy, IteratorTag)
{
    BOOST_STATIC_ASSERT(hpx::parallel::is_execution_policy<ExPolicy>::value);

    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<std::size_t> c(10007);
    std::iota(boost::begin(c), boost::end(c), std::rand());

    bool caught_exception = false;
    try {
        hpx::parallel::transform_reduce(policy,
            iterator(boost::begin(c)), iterator(boost::end(c)),
            std::size_t(42),
            [](std::size_t v1, std::size_t v2) {
                throw std::bad_alloc();
                return v1 + v2;
            },
            [](std::size_t v){return v;}
        );

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
void test_transform_reduce_bad_alloc(hpx::parallel::task_execution_policy, IteratorTag)
{
    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<std::size_t> c(10007);
    std::iota(boost::begin(c), boost::end(c), std::rand());

    bool caught_exception = false;
    try {
        hpx::future<void> f =
            hpx::parallel::transform_reduce(hpx::parallel::task,
                iterator(boost::begin(c)), iterator(boost::end(c)),
                std::size_t(42),
                [](std::size_t v1, std::size_t v2) {
                    throw std::bad_alloc();
                    return v1 + v2;
                },
                [](std::size_t v){return v;}
        );
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
void test_transform_reduce_bad_alloc()
{
    using namespace hpx::parallel;
    //If the execution policy object is of type vector_execution_policy,
    //  std::terminate shall be called. therefore we do not test exceptions
    //  with a vector execution policy
    test_transform_reduce_bad_alloc(seq, IteratorTag());
    test_transform_reduce_bad_alloc(par, IteratorTag());
    test_transform_reduce_bad_alloc(task, IteratorTag());

    test_transform_reduce_bad_alloc(execution_policy(seq), IteratorTag());
    test_transform_reduce_bad_alloc(execution_policy(par), IteratorTag());
    test_transform_reduce_bad_alloc(execution_policy(task), IteratorTag());
}

void transform_reduce_bad_alloc_test()
{
    test_transform_reduce_bad_alloc<std::random_access_iterator_tag>();
    test_transform_reduce_bad_alloc<std::forward_iterator_tag>();
    test_transform_reduce_bad_alloc<std::input_iterator_tag>();
}

int hpx_main()
{
    transform_reduce_test();
    transform_reduce_bad_alloc_test();
    transform_reduce_exception_test();

    return hpx::finalize();
}

int main(int argc, char* argv[])
{
    //By default run on all available cores
    std::vector<std::string> cfg;
    cfg.push_back("hpx.os_threads=" +
        boost::lexical_cast<std::string>(hpx::threads::hardware_concurrency()));

    HPX_TEST_EQ_MSG(hpx::init(argc, argv, cfg), 0,
        "HPX main exited with non-zero status");

    return hpx::util::report_errors();
}
