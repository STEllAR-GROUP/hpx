//  Copyright (c) 2014-2017 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>
#include <hpx/hpx.hpp>
#include <hpx/include/parallel_uninitialized_default_construct.hpp>
#include <hpx/util/lightweight_test.hpp>

#include <boost/atomic.hpp>
#include <boost/range/functions.hpp>

#include <cstddef>
#include <cstring>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#include "test_utils.hpp"

struct default_constructable
{
    default_constructable() : value_(42) {}
    int value_;
};

std::size_t const data_size = 10007;

////////////////////////////////////////////////////////////////////////////
template <typename ExPolicy, typename IteratorTag>
void test_uninitialized_default_construct_n(ExPolicy policy, IteratorTag)
{
    static_assert(
        hpx::parallel::execution::is_execution_policy<ExPolicy>::value,
        "hpx::parallel::execution::is_execution_policy<ExPolicy>::value");

    typedef default_constructable* base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    default_constructable* p = (default_constructable*)std::malloc(
        data_size * sizeof(default_constructable));
    std::memset(p, 0xcd, data_size * sizeof(default_constructable));

    hpx::parallel::uninitialized_default_construct_n(policy,
        iterator(p), data_size);

    std::size_t count = 0;
    std::for_each(p, p + data_size,
        [&count](default_constructable v1)
        {
            HPX_TEST_EQ(v1.value_, 42);
            ++count;
        });
    HPX_TEST_EQ(count, data_size);

    std::free(p);
}

template <typename ExPolicy, typename IteratorTag>
void test_uninitialized_default_construct_n_async(ExPolicy policy, IteratorTag)
{
    typedef default_constructable* base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    default_constructable* p =
        (default_constructable*)std::malloc(data_size * sizeof(std::size_t));
    std::memset(p, 0, data_size * sizeof(default_constructable));

    auto f =
        hpx::parallel::uninitialized_default_construct_n(policy,
            iterator(p), data_size);
    f.wait();

    std::size_t count = 0;
    std::for_each(p, p + data_size,
        [&count](default_constructable v1)
        {
            HPX_TEST_EQ(v1.value_, 42);
            ++count;
        });
    HPX_TEST_EQ(count, data_size);

    std::free(p);
}

template <typename IteratorTag>
void test_uninitialized_default_construct_n()
{
    using namespace hpx::parallel;

    test_uninitialized_default_construct_n(execution::seq, IteratorTag());
    test_uninitialized_default_construct_n(execution::par, IteratorTag());
    test_uninitialized_default_construct_n(execution::par_unseq, IteratorTag());

    test_uninitialized_default_construct_n_async(execution::seq(execution::task),
        IteratorTag());
    test_uninitialized_default_construct_n_async(execution::par(execution::task),
        IteratorTag());

#if defined(HPX_HAVE_GENERIC_EXECUTION_POLICY)
    test_uninitialized_default_construct_n(
        execution_policy(execution::seq), IteratorTag());
    test_uninitialized_default_construct_n(
        execution_policy(execution::par), IteratorTag());
    test_uninitialized_default_construct_n(
    execution_policy(execution::par_unseq),
        IteratorTag());

    test_uninitialized_default_construct_n(
        execution_policy(execution::seq(execution::task)), IteratorTag());
    test_uninitialized_default_construct_n(
        execution_policy(execution::par(execution::task)), IteratorTag());
#endif
}

void uninitialized_default_construct_n_test()
{
    test_uninitialized_default_construct_n<std::random_access_iterator_tag>();
    test_uninitialized_default_construct_n<std::forward_iterator_tag>();
}

///////////////////////////////////////////////////////////////////////////////
template<typename ExPolicy, typename IteratorTag>
void test_uninitialized_default_construct_n_exception(ExPolicy policy, IteratorTag)
{
    static_assert(
        hpx::parallel::execution::is_execution_policy<ExPolicy>::value,
        "hpx::parallel::execution::is_execution_policy<ExPolicy>::value");

    typedef test::count_instances_v<default_constructable> data_type;
    typedef data_type* base_iterator;
    typedef test::decorated_iterator<base_iterator, IteratorTag>
        decorated_iterator;

    data_type* p = (data_type*)std::malloc(data_size * sizeof(data_type));
    std::memset(p, 0xcd, data_size * sizeof(data_type));

    boost::atomic<std::size_t> throw_after(std::rand() % data_size); //-V104
    std::size_t throw_after_ = throw_after.load();

    data_type::instance_count.store(0);
    data_type::max_instance_count.store(0);

    bool caught_exception = false;
    try {
        hpx::parallel::uninitialized_default_construct_n(policy,
            decorated_iterator(
                p,
                [&throw_after]()
                {
                    if (throw_after-- == 0)
                        throw std::runtime_error("test");
                }),
            data_size);
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
    HPX_TEST_EQ(test::count_instances::instance_count.load(), std::size_t(0));
    HPX_TEST_LTE(throw_after_, data_type::max_instance_count.load());

    std::free(p);
}

template <typename ExPolicy, typename IteratorTag>
void test_uninitialized_default_construct_n_exception_async(
    ExPolicy policy, IteratorTag)
{
    typedef test::count_instances_v<default_constructable> data_type;
    typedef data_type* base_iterator;
    typedef test::decorated_iterator<base_iterator, IteratorTag>
        decorated_iterator;

    data_type* p = (data_type*)std::malloc(data_size * sizeof(data_type));
    std::memset(p, 0xcd, data_size * sizeof(data_type));

    boost::atomic<std::size_t> throw_after(std::rand() % data_size); //-V104
    std::size_t throw_after_ = throw_after.load();

    data_type::instance_count.store(0);
    data_type::max_instance_count.store(0);

    bool caught_exception = false;
    bool returned_from_algorithm = false;
    try {
        auto f =
            hpx::parallel::uninitialized_default_construct_n(policy,
                decorated_iterator(
                    p,
                    [&throw_after]()
                    {
                        if (throw_after-- == 0)
                            throw std::runtime_error("test");
                    }),
                data_size);

        returned_from_algorithm = true;
        f.get();

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
    HPX_TEST(returned_from_algorithm);
    HPX_TEST_EQ(test::count_instances::instance_count.load(), std::size_t(0));
    HPX_TEST_LTE(throw_after_, data_type::max_instance_count.load());

    std::free(p);
}

template<typename IteratorTag>
void test_uninitialized_default_construct_n_exception()
{
    using namespace hpx::parallel;

    // If the execution policy object is of type vector_execution_policy,
    // std::terminate shall be called. therefore we do not test exceptions
    // with a vector execution policy
    test_uninitialized_default_construct_n_exception(execution::seq, IteratorTag());
    test_uninitialized_default_construct_n_exception(execution::par, IteratorTag());

    test_uninitialized_default_construct_n_exception_async(
        execution::seq(execution::task),
        IteratorTag());
    test_uninitialized_default_construct_n_exception_async(
        execution::par(execution::task),
        IteratorTag());

#if defined(HPX_HAVE_GENERIC_EXECUTION_POLICY)
    test_uninitialized_default_construct_n_exception(
        execution_policy(execution::seq),
        IteratorTag());
    test_uninitialized_default_construct_n_exception(
        execution_policy(execution::par),
        IteratorTag());

    test_uninitialized_default_construct_n_exception(
        execution_policy(execution::seq(execution::task)),
        IteratorTag());
    test_uninitialized_default_construct_n_exception(
        execution_policy(execution::par(execution::task)),
        IteratorTag());
#endif
}

void uninitialized_default_construct_n_exception_test()
{
    test_uninitialized_default_construct_n_exception<std::random_access_iterator_tag>();
    test_uninitialized_default_construct_n_exception<std::forward_iterator_tag>();
}

////////////////////////////////////////////////////////////////////////////////
template< typename ExPolicy, typename IteratorTag>
void test_uninitialized_default_construct_n_bad_alloc(ExPolicy policy, IteratorTag)
{
    static_assert(
        hpx::parallel::execution::is_execution_policy<ExPolicy>::value,
        "hpx::parallel::execution::is_execution_policy<ExPolicy>::value");

    typedef test::count_instances_v<default_constructable> data_type;
    typedef data_type* base_iterator;
    typedef test::decorated_iterator<base_iterator, IteratorTag>
        decorated_iterator;

    data_type* p = (data_type*)std::malloc(data_size * sizeof(data_type));
    std::memset(p, 0xcd, data_size * sizeof(data_type));

    boost::atomic<std::size_t> throw_after(std::rand() % data_size); //-V104
    std::size_t throw_after_ = throw_after.load();

    data_type::instance_count.store(0);
    data_type::max_instance_count.store(0);

    bool caught_bad_alloc = false;
    try {
        hpx::parallel::uninitialized_default_construct_n(policy,
            decorated_iterator(
                p,
                [&throw_after]()
                {
                    if (throw_after-- == 0)
                        throw std::bad_alloc();
                }),
            data_size);

        HPX_TEST(false);
    }
    catch(std::bad_alloc const&) {
        caught_bad_alloc = true;
    }
    catch(...) {
        HPX_TEST(false);
    }

    HPX_TEST(caught_bad_alloc);
    HPX_TEST_EQ(test::count_instances::instance_count.load(), std::size_t(0));
    HPX_TEST_LTE(throw_after_, data_type::max_instance_count.load());

    std::free(p);
}

template <typename ExPolicy, typename IteratorTag>
void test_uninitialized_default_construct_n_bad_alloc_async(
    ExPolicy policy, IteratorTag)
{
    typedef test::count_instances_v<default_constructable> data_type;
    typedef data_type* base_iterator;
    typedef test::decorated_iterator<base_iterator, IteratorTag>
        decorated_iterator;

    data_type* p = (data_type*)std::malloc(data_size * sizeof(data_type));
    std::memset(p, 0xcd, data_size * sizeof(data_type));

    boost::atomic<std::size_t> throw_after(std::rand() % data_size); //-V104
    std::size_t throw_after_ = throw_after.load();

    data_type::instance_count.store(0);
    data_type::max_instance_count.store(0);

    bool caught_bad_alloc = false;
    bool returned_from_algorithm = false;
    try {
        auto f =
            hpx::parallel::uninitialized_default_construct_n(policy,
                decorated_iterator(
                    p,
                    [&throw_after]()
                    {
                        if (throw_after-- == 0)
                            throw std::bad_alloc();
                    }),
                data_size);

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
    HPX_TEST_EQ(test::count_instances::instance_count.load(), std::size_t(0));
    HPX_TEST_LTE(throw_after_, data_type::max_instance_count.load());

    std::free(p);
}

template<typename IteratorTag>
void test_uninitialized_default_construct_n_bad_alloc()
{
    using namespace hpx::parallel;

    // If the execution policy object is of type vector_execution_policy,
    // std::terminate shall be called. therefore we do not test exceptions
    // with a vector execution policy
    test_uninitialized_default_construct_n_bad_alloc(execution::seq, IteratorTag());
    test_uninitialized_default_construct_n_bad_alloc(execution::par, IteratorTag());

    test_uninitialized_default_construct_n_bad_alloc_async(
        execution::seq(execution::task),
        IteratorTag());
    test_uninitialized_default_construct_n_bad_alloc_async(
        execution::par(execution::task),
        IteratorTag());

#if defined(HPX_HAVE_GENERIC_EXECUTION_POLICY)
    test_uninitialized_default_construct_n_bad_alloc(
        execution_policy(execution::seq),
        IteratorTag());
    test_uninitialized_default_construct_n_bad_alloc(
        execution_policy(execution::par),
        IteratorTag());

    test_uninitialized_default_construct_n_bad_alloc(
        execution_policy(execution::seq(execution::task)),
        IteratorTag());
    test_uninitialized_default_construct_n_bad_alloc(
        execution_policy(execution::par(execution::task)),
        IteratorTag());
#endif
}

void uninitialized_default_construct_n_bad_alloc_test()
{
    test_uninitialized_default_construct_n_bad_alloc<std::random_access_iterator_tag>();
    test_uninitialized_default_construct_n_bad_alloc<std::forward_iterator_tag>();
}

int hpx_main(boost::program_options::variables_map& vm)
{
    unsigned int seed = (unsigned int)std::time(nullptr);
    if (vm.count("seed"))
        seed = vm["seed"].as<unsigned int>();

    std::cout << "using seed: " << seed << std::endl;
    std::srand(seed);

    uninitialized_default_construct_n_test();
    uninitialized_default_construct_n_exception_test();
    uninitialized_default_construct_n_bad_alloc_test();
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
