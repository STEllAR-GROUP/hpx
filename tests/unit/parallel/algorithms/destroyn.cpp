//  Copyright (c) 2014-2017 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>
#include <hpx/hpx.hpp>
#include <hpx/include/parallel_destroy.hpp>
#include <hpx/util/lightweight_test.hpp>

#include <boost/atomic.hpp>
#include <boost/range/functions.hpp>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include "test_utils.hpp"

boost::atomic<std::size_t> destruct_count(0);

struct destructable
{
    destructable()
      : value_(0)
    {}

    ~destructable()
    {
        ++destruct_count;
    }

    std::uint32_t value_;
};

std::size_t const data_size = 10007;

////////////////////////////////////////////////////////////////////////////
template <typename ExPolicy, typename IteratorTag>
void test_destroy_n(ExPolicy policy, IteratorTag)
{
    static_assert(
        hpx::parallel::execution::is_execution_policy<ExPolicy>::value,
        "hpx::parallel::execution::is_execution_policy<ExPolicy>::value");

    typedef destructable* base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    destructable* p = (destructable*)std::malloc(
        data_size * sizeof(destructable));

    // value-initialize data in array
    std::for_each(
        p, p + data_size,
        [](destructable& d)
        {
            ::new (static_cast<void*>(std::addressof(d))) destructable;
        });

    destruct_count.store(0);

    hpx::parallel::destroy_n(
        std::forward<ExPolicy>(policy),
        iterator(p), data_size);

    HPX_TEST_EQ(destruct_count.load(), data_size);

    std::free(p);
}

template <typename ExPolicy, typename IteratorTag>
void test_destroy_n_async(ExPolicy policy, IteratorTag)
{
    typedef destructable* base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    destructable* p =
        (destructable*)std::malloc(data_size * sizeof(destructable));

    // value-initialize data in array
    std::for_each(
        p, p + data_size,
        [](destructable& d)
        {
            ::new (static_cast<void*>(std::addressof(d))) destructable;
        });

    destruct_count.store(0);

    auto f =
        hpx::parallel::destroy_n(
            std::forward<ExPolicy>(policy),
            iterator(p), data_size);
    f.wait();

    HPX_TEST_EQ(destruct_count.load(), data_size);

    std::free(p);
}

template <typename IteratorTag>
void test_destroy_n()
{
    using namespace hpx::parallel;

    test_destroy_n(execution::seq, IteratorTag());
    test_destroy_n(execution::par, IteratorTag());
    test_destroy_n(execution::par_unseq, IteratorTag());

    test_destroy_n_async(execution::seq(execution::task), IteratorTag());
    test_destroy_n_async(execution::par(execution::task), IteratorTag());

#if defined(HPX_HAVE_GENERIC_EXECUTION_POLICY)
    test_destroy_n(execution_policy(execution::seq), IteratorTag());
    test_destroy_n(execution_policy(execution::par), IteratorTag());
    test_destroy_n(execution_policy(execution::par_unseq), IteratorTag());

    test_destroy_n(
        execution_policy(execution::seq(execution::task)), IteratorTag());
    test_destroy_n(
        execution_policy(execution::par(execution::task)), IteratorTag());
#endif
}

void destroy_n_test()
{
    test_destroy_n<std::random_access_iterator_tag>();
    test_destroy_n<std::forward_iterator_tag>();
}

///////////////////////////////////////////////////////////////////////////////
template<typename ExPolicy, typename IteratorTag>
void test_destroy_n_exception(ExPolicy policy, IteratorTag)
{
    static_assert(
        hpx::parallel::execution::is_execution_policy<ExPolicy>::value,
        "hpx::parallel::execution::is_execution_policy<ExPolicy>::value");

    typedef test::count_instances_v<destructable> data_type;
    typedef data_type* base_iterator;
    typedef test::decorated_iterator<base_iterator, IteratorTag>
        decorated_iterator;

    data_type* p = (data_type*)std::malloc(data_size * sizeof(data_type));

    data_type::instance_count.store(0);
    data_type::max_instance_count.store(0);

    // value-initialize data in array
    std::for_each(
        p, p + data_size,
        [](data_type& d)
        {
            ::new (static_cast<void*>(std::addressof(d))) data_type;
        });

    HPX_TEST_EQ(data_type::instance_count.load(), data_size);

    boost::atomic<std::size_t> throw_after(std::rand() % data_size); //-V104
    std::size_t throw_after_ = throw_after.load();

    bool caught_exception = false;
    try {
        hpx::parallel::destroy_n(policy,
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
    HPX_TEST_LTE(data_type::instance_count.load(),
        std::size_t(data_size-throw_after_));

    std::free(p);
}

template <typename ExPolicy, typename IteratorTag>
void test_destroy_n_exception_async(
    ExPolicy policy, IteratorTag)
{
    typedef test::count_instances_v<destructable> data_type;
    typedef data_type* base_iterator;
    typedef test::decorated_iterator<base_iterator, IteratorTag>
        decorated_iterator;

    data_type* p = (data_type*)std::malloc(data_size * sizeof(data_type));

    data_type::instance_count.store(0);
    data_type::max_instance_count.store(0);

    // value-initialize data in array
    std::for_each(
        p, p + data_size,
        [](data_type& d)
        {
            ::new (static_cast<void*>(std::addressof(d))) data_type;
        });

    HPX_TEST_EQ(data_type::instance_count.load(), data_size);

    boost::atomic<std::size_t> throw_after(std::rand() % data_size); //-V104
    std::size_t throw_after_ = throw_after.load();

    bool caught_exception = false;
    bool returned_from_algorithm = false;
    try {
        auto f =
            hpx::parallel::destroy_n(policy,
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
    HPX_TEST_LTE(data_type::instance_count.load(),
        std::size_t(data_size-throw_after_));

    std::free(p);
}

template<typename IteratorTag>
void test_destroy_n_exception()
{
    using namespace hpx::parallel;

    // If the execution policy object is of type vector_execution_policy,
    // std::terminate shall be called. therefore we do not test exceptions
    // with a vector execution policy
    test_destroy_n_exception(execution::seq, IteratorTag());
    test_destroy_n_exception(execution::par, IteratorTag());

    test_destroy_n_exception_async(execution::seq(execution::task), IteratorTag());
    test_destroy_n_exception_async(execution::par(execution::task), IteratorTag());

#if defined(HPX_HAVE_GENERIC_EXECUTION_POLICY)
    test_destroy_n_exception(execution_policy(execution::seq), IteratorTag());
    test_destroy_n_exception(execution_policy(execution::par), IteratorTag());

    test_destroy_n_exception(execution_policy(execution::seq(execution::task)),
        IteratorTag());
    test_destroy_n_exception(execution_policy(execution::par(execution::task)),
        IteratorTag());
#endif
}

void destroy_n_exception_test()
{
    test_destroy_n_exception<std::random_access_iterator_tag>();
    test_destroy_n_exception<std::forward_iterator_tag>();
}

////////////////////////////////////////////////////////////////////////////////
template< typename ExPolicy, typename IteratorTag>
void test_destroy_n_bad_alloc(ExPolicy policy, IteratorTag)
{
    static_assert(
        hpx::parallel::execution::is_execution_policy<ExPolicy>::value,
        "hpx::parallel::execution::is_execution_policy<ExPolicy>::value");

    typedef test::count_instances_v<destructable> data_type;
    typedef data_type* base_iterator;
    typedef test::decorated_iterator<base_iterator, IteratorTag>
        decorated_iterator;

    data_type* p = (data_type*)std::malloc(data_size * sizeof(data_type));

    data_type::instance_count.store(0);
    data_type::max_instance_count.store(0);

    // value-initialize data in array
    std::for_each(
        p, p + data_size,
        [](data_type& d)
        {
            ::new (static_cast<void*>(std::addressof(d))) data_type;
        });

    HPX_TEST_EQ(data_type::instance_count.load(), data_size);

    boost::atomic<std::size_t> throw_after(std::rand() % data_size); //-V104
    std::size_t throw_after_ = throw_after.load();

    bool caught_bad_alloc = false;
    try {
        hpx::parallel::destroy_n(policy,
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
    HPX_TEST_LTE(data_type::instance_count.load(),
        std::size_t(data_size-throw_after_));

    std::free(p);
}

template <typename ExPolicy, typename IteratorTag>
void test_destroy_n_bad_alloc_async(
    ExPolicy policy, IteratorTag)
{
    typedef test::count_instances_v<destructable> data_type;
    typedef data_type* base_iterator;
    typedef test::decorated_iterator<base_iterator, IteratorTag>
        decorated_iterator;

    data_type* p = (data_type*)std::malloc(data_size * sizeof(data_type));

    data_type::instance_count.store(0);
    data_type::max_instance_count.store(0);

    // value-initialize data in array
    std::for_each(
        p, p + data_size,
        [](data_type& d)
        {
            ::new (static_cast<void*>(std::addressof(d))) data_type;
        });

    HPX_TEST_EQ(data_type::instance_count.load(), data_size);

    boost::atomic<std::size_t> throw_after(std::rand() % data_size); //-V104
    std::size_t throw_after_ = throw_after.load();

    bool caught_bad_alloc = false;
    bool returned_from_algorithm = false;
    try {
        auto f =
            hpx::parallel::destroy_n(policy,
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
    HPX_TEST_LTE(data_type::instance_count.load(),
        std::size_t(data_size-throw_after_));

    std::free(p);
}

template<typename IteratorTag>
void test_destroy_n_bad_alloc()
{
    using namespace hpx::parallel;

    // If the execution policy object is of type vector_execution_policy,
    // std::terminate shall be called. therefore we do not test exceptions
    // with a vector execution policy
    test_destroy_n_bad_alloc(execution::seq, IteratorTag());
    test_destroy_n_bad_alloc(execution::par, IteratorTag());

    test_destroy_n_bad_alloc_async(
        execution::seq(execution::task),
        IteratorTag());
    test_destroy_n_bad_alloc_async(
        execution::par(execution::task),
        IteratorTag());

#if defined(HPX_HAVE_GENERIC_EXECUTION_POLICY)
    test_destroy_n_bad_alloc(
        execution_policy(execution::seq),
        IteratorTag());
    test_destroy_n_bad_alloc(
        execution_policy(execution::par),
        IteratorTag());

    test_destroy_n_bad_alloc(
        execution_policy(execution::seq(execution::task)),
        IteratorTag());
    test_destroy_n_bad_alloc(
        execution_policy(execution::par(execution::task)),
        IteratorTag());
#endif
}

void destroy_n_bad_alloc_test()
{
    test_destroy_n_bad_alloc<std::random_access_iterator_tag>();
    test_destroy_n_bad_alloc<std::forward_iterator_tag>();
}

int hpx_main(boost::program_options::variables_map& vm)
{
    unsigned int seed = (unsigned int)std::time(nullptr);
    if (vm.count("seed"))
        seed = vm["seed"].as<unsigned int>();

    std::cout << "using seed: " << seed << std::endl;
    std::srand(seed);

    destroy_n_test();
    destroy_n_exception_test();
    destroy_n_bad_alloc_test();
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
