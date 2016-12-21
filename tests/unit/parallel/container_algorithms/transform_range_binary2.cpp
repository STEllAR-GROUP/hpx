//  Copyright (c) 2014-2015 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>
#include <hpx/hpx.hpp>
#include <hpx/include/parallel_transform.hpp>
#include <hpx/util/lightweight_test.hpp>

#include <boost/range/functions.hpp>

#include <cstddef>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#include "test_utils.hpp"

///////////////////////////////////////////////////////////////////////////////
template <typename ExPolicy, typename IteratorTag>
void test_transform_binary(ExPolicy policy, IteratorTag)
{
    BOOST_STATIC_ASSERT(hpx::parallel::execution::is_execution_policy<ExPolicy>::value);

    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    typedef test::test_container<std::vector<int>, IteratorTag> test_vector;

    test_vector c1(10007);
    test_vector c2(10007);
    std::vector<std::size_t> d1(c1.size()); //-V656
    std::iota(boost::begin(c1), boost::end(c1), std::rand());
    std::iota(boost::begin(c2), boost::end(c2), std::rand());

    auto add =
        [](std::size_t v1, std::size_t v2) {
            return v1 + v2;
        };

    auto result =
        hpx::parallel::transform(policy,
            c1, c2, boost::begin(d1), add);

    HPX_TEST(hpx::util::get<0>(result) == boost::end(c1));
    HPX_TEST(hpx::util::get<1>(result) == boost::end(c2));
    HPX_TEST(hpx::util::get<2>(result) == boost::end(d1));

    // verify values
    std::vector<std::size_t> d2(c1.size());
    std::transform(boost::begin(c1), boost::end(c1),
        boost::begin(c2), boost::begin(d2), add);

    std::size_t count = 0;
    HPX_TEST(std::equal(boost::begin(d1), boost::end(d1), boost::begin(d2),
        [&count](std::size_t v1, std::size_t v2) -> bool {
            HPX_TEST_EQ(v1, v2);
            ++count;
            return v1 == v2;
        }));
    HPX_TEST_EQ(count, d2.size());
}

template <typename ExPolicy, typename IteratorTag>
void test_transform_binary_async(ExPolicy p, IteratorTag)
{
    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    typedef test::test_container<std::vector<int>, IteratorTag> test_vector;

    test_vector c1(10007);
    test_vector c2(10007);
    std::vector<std::size_t> d1(c1.size()); //-V656
    std::iota(boost::begin(c1), boost::end(c1), std::rand());
    std::iota(boost::begin(c2), boost::end(c2), std::rand());

    auto add =
        [](std::size_t v1, std::size_t v2) {
            return v1 + v2;
        };

    auto f =
        hpx::parallel::transform(p,
            c1, c2, boost::begin(d1), add);
    f.wait();

    auto result = f.get();
    HPX_TEST(hpx::util::get<0>(result) == boost::end(c1));
    HPX_TEST(hpx::util::get<1>(result) == boost::end(c2));
    HPX_TEST(hpx::util::get<2>(result) == boost::end(d1));

    // verify values
    std::vector<std::size_t> d2(c1.size());
    std::transform(boost::begin(c1), boost::end(c1),
        boost::begin(c2), boost::begin(d2), add);

    std::size_t count = 0;
    HPX_TEST(std::equal(boost::begin(d1), boost::end(d1), boost::begin(d2),
        [&count](std::size_t v1, std::size_t v2) -> bool {
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

    test_transform_binary(execution::seq, IteratorTag());
    test_transform_binary(execution::par, IteratorTag());
    test_transform_binary(execution::par_unseq, IteratorTag());

    test_transform_binary_async(execution::seq(execution::task), IteratorTag());
    test_transform_binary_async(execution::par(execution::task), IteratorTag());

#if defined(HPX_HAVE_GENERIC_EXECUTION_POLICY)
    test_transform_binary(execution_policy(execution::seq), IteratorTag());
    test_transform_binary(execution_policy(execution::par), IteratorTag());
    test_transform_binary(execution_policy(execution::par_unseq), IteratorTag());

    test_transform_binary(execution_policy(execution::seq(execution::task)), IteratorTag());
    test_transform_binary(execution_policy(execution::par(execution::task)), IteratorTag());
#endif
}

void transform_binary_test()
{
    test_transform_binary<std::random_access_iterator_tag>();
    test_transform_binary<std::forward_iterator_tag>();
    test_transform_binary<std::input_iterator_tag>();
}

///////////////////////////////////////////////////////////////////////////////
template <typename ExPolicy, typename IteratorTag>
void test_transform_binary_exception(ExPolicy policy, IteratorTag)
{
    BOOST_STATIC_ASSERT(hpx::parallel::execution::is_execution_policy<ExPolicy>::value);

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
            boost::make_iterator_range(
                iterator(boost::begin(c1)), iterator(boost::end(c1))
            ),
            boost::make_iterator_range(
                boost::begin(c2), boost::end(c2)
            ),
            boost::begin(d1),
            [](std::size_t v1, std::size_t v2) {
                return throw std::runtime_error("test"), v1 + v2;
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

template <typename ExPolicy, typename IteratorTag>
void test_transform_binary_exception_async(ExPolicy p, IteratorTag)
{
    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<std::size_t> c1(10007);
    std::vector<std::size_t> c2(c1.size());
    std::vector<std::size_t> d1(c1.size()); //-V656
    std::iota(boost::begin(c1), boost::end(c1), std::rand());
    std::iota(boost::begin(c2), boost::end(c2), std::rand());

    bool caught_exception = false;
    bool returned_from_algorithm = false;
    try {
        auto f =
            hpx::parallel::transform(p,
                boost::make_iterator_range(
                    iterator(boost::begin(c1)), iterator(boost::end(c1))
                ),
                boost::make_iterator_range(
                    boost::begin(c2), boost::end(c2)
                ),
                boost::begin(d1),
                [](std::size_t v1, std::size_t v2) {
                    return throw std::runtime_error("test"), v1 + v2;
                });
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
void test_transform_binary_exception()
{
    using namespace hpx::parallel;

    // If the execution policy object is of type vector_execution_policy,
    // std::terminate shall be called. therefore we do not test exceptions
    // with a vector execution policy
    test_transform_binary_exception(execution::seq, IteratorTag());
    test_transform_binary_exception(execution::par, IteratorTag());

    test_transform_binary_exception_async(execution::seq(execution::task), IteratorTag());
    test_transform_binary_exception_async(execution::par(execution::task), IteratorTag());

#if defined(HPX_HAVE_GENERIC_EXECUTION_POLICY)
    test_transform_binary_exception(execution_policy(execution::seq),
        IteratorTag());
    test_transform_binary_exception(execution_policy(execution::par),
        IteratorTag());

    test_transform_binary_exception(execution_policy(execution::seq(execution::task)),
        IteratorTag());
    test_transform_binary_exception(execution_policy(execution::par(execution::task)),
        IteratorTag());
#endif
}

void transform_binary_exception_test()
{
    test_transform_binary_exception<std::random_access_iterator_tag>();
    test_transform_binary_exception<std::forward_iterator_tag>();
    test_transform_binary_exception<std::input_iterator_tag>();
}

///////////////////////////////////////////////////////////////////////////////
template <typename ExPolicy, typename IteratorTag>
void test_transform_binary_bad_alloc(ExPolicy policy, IteratorTag)
{
    BOOST_STATIC_ASSERT(hpx::parallel::execution::is_execution_policy<ExPolicy>::value);

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
            boost::make_iterator_range(
                iterator(boost::begin(c1)), iterator(boost::end(c1))
            ),
            boost::make_iterator_range(
                boost::begin(c2), boost::end(c2)
            ),
            boost::begin(d1),
            [](std::size_t v1, std::size_t v2) {
                return throw std::bad_alloc(), v1 + v2;
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

template <typename ExPolicy, typename IteratorTag>
void test_transform_binary_bad_alloc_async(ExPolicy p, IteratorTag)
{
    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<std::size_t> c1(10007);
    std::vector<std::size_t> c2(c1.size());
    std::vector<std::size_t> d1(c1.size()); //-V656
    std::iota(boost::begin(c1), boost::end(c1), std::rand());
    std::iota(boost::begin(c2), boost::end(c2), std::rand());

    bool caught_bad_alloc = false;
    bool returned_from_algorithm = false;
    try {
        auto f =
            hpx::parallel::transform(p,
                boost::make_iterator_range(
                    iterator(boost::begin(c1)), iterator(boost::end(c1))
                ),
                boost::make_iterator_range(
                    boost::begin(c2), boost::end(c2)
                ),
                boost::begin(d1),
                [](std::size_t v1, std::size_t v2) {
                    return throw std::bad_alloc(), v1 + v2;
                });
        returned_from_algorithm = true;
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
    HPX_TEST(returned_from_algorithm);
}

template <typename IteratorTag>
void test_transform_binary_bad_alloc()
{
    using namespace hpx::parallel;

    // If the execution policy object is of type vector_execution_policy,
    // std::terminate shall be called. therefore we do not test exceptions
    // with a vector execution policy
    test_transform_binary_bad_alloc(execution::seq, IteratorTag());
    test_transform_binary_bad_alloc(execution::par, IteratorTag());

    test_transform_binary_bad_alloc_async(execution::seq(execution::task), IteratorTag());
    test_transform_binary_bad_alloc_async(execution::par(execution::task), IteratorTag());

#if defined(HPX_HAVE_GENERIC_EXECUTION_POLICY)
    test_transform_binary_bad_alloc(execution_policy(execution::seq),
        IteratorTag());
    test_transform_binary_bad_alloc(execution_policy(execution::par),
        IteratorTag());

    test_transform_binary_bad_alloc(execution_policy(execution::seq(execution::task)),
        IteratorTag());
    test_transform_binary_bad_alloc(execution_policy(execution::par(execution::task)),
        IteratorTag());
#endif
}

void transform_binary_bad_alloc_test()
{
    test_transform_binary_bad_alloc<std::random_access_iterator_tag>();
    test_transform_binary_bad_alloc<std::forward_iterator_tag>();
    test_transform_binary_bad_alloc<std::input_iterator_tag>();
}

///////////////////////////////////////////////////////////////////////////////
int hpx_main(boost::program_options::variables_map& vm)
{
    unsigned int seed = (unsigned int)std::time(nullptr);
    if (vm.count("seed"))
        seed = vm["seed"].as<unsigned int>();

    std::cout << "using seed: " << seed << std::endl;
    std::srand(seed);

    transform_binary_test();
    transform_binary_exception_test();
    transform_binary_bad_alloc_test();
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
    // By default this test should run on all available cores
    std::vector<std::string> const cfg = {
        "hpx.os_threads=all"
    };

    // Initialize and run HPX
    HPX_TEST_EQ_MSG(hpx::init(desc_commandline, argc, argv, cfg), 0,
        "HPX main exited with non-zero status");

    return hpx::util::report_errors();
}


