//  Copyright (c) 2014 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>
#include <hpx/hpx.hpp>
#include <hpx/include/parallel_transform_reduce.hpp>
#include <hpx/util/lightweight_test.hpp>

#include <boost/range/functions.hpp>

#include <algorithm>
#include <cstddef>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#include "test_utils.hpp"

///////////////////////////////////////////////////////////////////////////////
template <typename ExPolicy, typename IteratorTag>
void test_transform_reduce(ExPolicy policy, IteratorTag)
{
    static_assert(
        hpx::parallel::execution::is_execution_policy<ExPolicy>::value,
        "hpx::parallel::execution::is_execution_policy<ExPolicy>::value");

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

template <typename ExPolicy, typename IteratorTag>
void test_transform_reduce_async(ExPolicy p, IteratorTag)
{
    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<std::size_t> c(10007);
    std::iota(boost::begin(c), boost::end(c), std::rand());

    std::size_t val(42);
    auto op =
        [val](std::size_t v1, std::size_t v2) {
            return v1 + v2 + val;
        };

    hpx::future<std::size_t> f =
        hpx::parallel::transform_reduce(p,
            iterator(boost::begin(c)), iterator(boost::end(c)),
            val, op, [](std::size_t v){ return v; });
    f.wait();

    // verify values
    std::size_t r2 = std::accumulate(boost::begin(c), boost::end(c), val, op);
    HPX_TEST_EQ(f.get(), r2);
}

template <typename IteratorTag>
void test_transform_reduce()
{
    using namespace hpx::parallel;

    test_transform_reduce(execution::seq, IteratorTag());
    test_transform_reduce(execution::par, IteratorTag());
    test_transform_reduce(execution::par_unseq, IteratorTag());

    test_transform_reduce_async(execution::seq(execution::task), IteratorTag());
    test_transform_reduce_async(execution::par(execution::task), IteratorTag());

#if defined(HPX_HAVE_GENERIC_EXECUTION_POLICY)
    test_transform_reduce(execution_policy(execution::seq), IteratorTag());
    test_transform_reduce(execution_policy(execution::par), IteratorTag());
    test_transform_reduce(execution_policy(execution::par_unseq), IteratorTag());

    test_transform_reduce(execution_policy(execution::seq(execution::task)),
        IteratorTag());
    test_transform_reduce(execution_policy(execution::par(execution::task)),
        IteratorTag());
#endif
}

void transform_reduce_test()
{
    test_transform_reduce<std::random_access_iterator_tag>();
    test_transform_reduce<std::forward_iterator_tag>();
#if defined(HPX_HAVE_ALGORITHM_INPUT_ITERATOR_SUPPORT)
    test_transform_reduce<std::input_iterator_tag>();
#endif
}

///////////////////////////////////////////////////////////////////////////////
template <typename ExPolicy, typename IteratorTag>
void test_transform_reduce_exception(ExPolicy policy, IteratorTag)
{
    static_assert(
        hpx::parallel::execution::is_execution_policy<ExPolicy>::value,
        "hpx::parallel::execution::is_execution_policy<ExPolicy>::value");

    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<std::size_t> c(10007);
    std::iota(boost::begin(c), boost::end(c), std::rand());

    bool caught_exception = false;
    try {
        hpx::parallel::transform_reduce(policy,
            iterator(boost::begin(c)), iterator(boost::end(c)),
            std::size_t(42),
            [](std::size_t v1, std::size_t v2)
            {
                return throw std::runtime_error("test"), v1 + v2;
            },
            [](std::size_t v){ return v; }
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

template <typename ExPolicy, typename IteratorTag>
void test_transform_reduce_exception_async(ExPolicy p, IteratorTag)
{
    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<std::size_t> c(10007);
    std::iota(boost::begin(c), boost::end(c), std::rand());

    bool caught_exception = false;
    bool returned_from_algorithm = false;
    try {
        hpx::future<void> f =
            hpx::parallel::transform_reduce(p,
                iterator(boost::begin(c)), iterator(boost::end(c)),
                std::size_t(42),
                [](std::size_t v1, std::size_t v2)
                {
                    return throw std::runtime_error("test"), v1 + v2;
                },
                [](std::size_t v){ return v; }
            );
        returned_from_algorithm = true;
        f.get();

        HPX_TEST(false);
    }
    catch(hpx::exception_list const& e) {
        caught_exception = true;
        test::test_num_exceptions<ExPolicy, IteratorTag>::call(p, e);
    }
    catch(...) {
        HPX_TEST(false);
    }

    HPX_TEST(caught_exception);
    HPX_TEST(returned_from_algorithm);
}

template <typename IteratorTag>
void test_transform_reduce_exception()
{
    using namespace hpx::parallel;

    // If the execution policy object is of type vector_execution_policy,
    // std::terminate shall be called. therefore we do not test exceptions
    // with a vector execution policy
    test_transform_reduce_exception(execution::seq, IteratorTag());
    test_transform_reduce_exception(execution::par, IteratorTag());

    test_transform_reduce_exception_async(execution::seq(execution::task),
        IteratorTag());
    test_transform_reduce_exception_async(execution::par(execution::task),
        IteratorTag());

#if defined(HPX_HAVE_GENERIC_EXECUTION_POLICY)
    test_transform_reduce_exception(execution_policy(execution::seq),
        IteratorTag());
    test_transform_reduce_exception(execution_policy(execution::par),
        IteratorTag());

    test_transform_reduce_exception(
        execution_policy(execution::seq(execution::task)),
        IteratorTag());
    test_transform_reduce_exception(
        execution_policy(execution::par(execution::task)),
        IteratorTag());
#endif
}

void transform_reduce_exception_test()
{
    test_transform_reduce_exception<std::random_access_iterator_tag>();
    test_transform_reduce_exception<std::forward_iterator_tag>();
#if defined(HPX_HAVE_ALGORITHM_INPUT_ITERATOR_SUPPORT)
    test_transform_reduce_exception<std::input_iterator_tag>();
#endif
}

///////////////////////////////////////////////////////////////////////////////
template <typename ExPolicy, typename IteratorTag>
void test_transform_reduce_bad_alloc(ExPolicy policy, IteratorTag)
{
    static_assert(
        hpx::parallel::execution::is_execution_policy<ExPolicy>::value,
        "hpx::parallel::execution::is_execution_policy<ExPolicy>::value");

    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<std::size_t> c(10007);
    std::iota(boost::begin(c), boost::end(c), std::rand());

    bool caught_exception = false;
    try {
        hpx::parallel::transform_reduce(policy,
            iterator(boost::begin(c)), iterator(boost::end(c)),
            std::size_t(42),
            [](std::size_t v1, std::size_t v2)
            {
                return throw std::bad_alloc(), v1 + v2;
            },
            [](std::size_t v){ return v; }
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

template <typename ExPolicy, typename IteratorTag>
void test_transform_reduce_bad_alloc_async(ExPolicy p, IteratorTag)
{
    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<std::size_t> c(10007);
    std::iota(boost::begin(c), boost::end(c), std::rand());

    bool caught_exception = false;
    bool returned_from_algorithm = false;
    try {
        hpx::future<void> f =
            hpx::parallel::transform_reduce(p,
                iterator(boost::begin(c)), iterator(boost::end(c)),
                std::size_t(42),
                [](std::size_t v1, std::size_t v2)
                {
                    return throw std::bad_alloc(), v1 + v2;
                },
                [](std::size_t v){ return v; }
        );
        returned_from_algorithm = true;
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
    HPX_TEST(returned_from_algorithm);
}

template <typename IteratorTag>
void test_transform_reduce_bad_alloc()
{
    using namespace hpx::parallel;

    // If the execution policy object is of type vector_execution_policy,
    // std::terminate shall be called. therefore we do not test exceptions
    // with a vector execution policy
    test_transform_reduce_bad_alloc(execution::seq, IteratorTag());
    test_transform_reduce_bad_alloc(execution::par, IteratorTag());

    test_transform_reduce_bad_alloc_async(execution::seq(execution::task),
        IteratorTag());
    test_transform_reduce_bad_alloc_async(execution::par(execution::task),
        IteratorTag());

#if defined(HPX_HAVE_GENERIC_EXECUTION_POLICY)
    test_transform_reduce_bad_alloc(execution_policy(execution::seq),
        IteratorTag());
    test_transform_reduce_bad_alloc(execution_policy(execution::par),
        IteratorTag());

    test_transform_reduce_bad_alloc(
        execution_policy(execution::seq(execution::task)),
        IteratorTag());
    test_transform_reduce_bad_alloc(
        execution_policy(execution::par(execution::task)),
        IteratorTag());
#endif
}

void transform_reduce_bad_alloc_test()
{
    test_transform_reduce_bad_alloc<std::random_access_iterator_tag>();
    test_transform_reduce_bad_alloc<std::forward_iterator_tag>();
#if defined(HPX_HAVE_ALGORITHM_INPUT_ITERATOR_SUPPORT)
    test_transform_reduce_bad_alloc<std::input_iterator_tag>();
#endif
}

int hpx_main(boost::program_options::variables_map& vm)
{
    unsigned int seed = (unsigned int)std::time(nullptr);
    if (vm.count("seed"))
        seed = vm["seed"].as<unsigned int>();

    std::cout << "using seed: " << seed << std::endl;
    std::srand(seed);

    transform_reduce_test();
    transform_reduce_bad_alloc_test();
    transform_reduce_exception_test();

    return hpx::finalize();
}

int main(int argc, char* argv[])
{
    // add command line option which controls the random number generator seed
    using namespace boost::program_options;
    options_description desc_commandline(
        "Usage: " HPX_APPLICATION_STRING " [options]");

    desc_commandline.add_options()
        ("seed,s", value<unsigned int>(),
        "the random number generator seed to use for this run")
        ;

    //By default run on all available cores
    std::vector<std::string> const cfg = {
        "hpx.os_threads=all"
    };

    // Initialize and run HPX
    HPX_TEST_EQ_MSG(hpx::init(desc_commandline, argc, argv, cfg), 0,
        "HPX main exited with non-zero status");

    return hpx::util::report_errors();
}
