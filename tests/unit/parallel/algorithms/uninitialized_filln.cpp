//  Copyright (c) 2014 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>
#include <hpx/hpx.hpp>
#include <hpx/include/parallel_uninitialized_fill.hpp>
#include <hpx/util/lightweight_test.hpp>

#include <boost/range/functions.hpp>

#include <string>

#include "test_utils.hpp"

////////////////////////////////////////////////////////////////////////////
template <typename ExPolicy, typename IteratorTag>
void test_uninitialized_fill_n(ExPolicy policy, IteratorTag)
{
    static_assert(
        hpx::parallel::is_execution_policy<ExPolicy>::value,
        "hpx::parallel::is_execution_policy<ExPolicy>::value");

    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<std::size_t> c(10007);
    std::iota(boost::begin(c), boost::end(c), std::rand());

    hpx::parallel::uninitialized_fill_n(policy,
        iterator(boost::begin(c)), c.size(), 10);

    // verify values
    std::size_t count = 0;
    std::for_each(boost::begin(c), boost::end(c),
        [&count](std::size_t v) -> void {
            HPX_TEST_EQ(v, std::size_t(10));
            ++count;
        });
    HPX_TEST_EQ(count, c.size());
}

template <typename ExPolicy, typename IteratorTag>
void test_uninitialized_fill_n_async(ExPolicy p, IteratorTag)
{
    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<std::size_t> c(10007);
    std::iota(boost::begin(c), boost::end(c), std::rand());

    hpx::future<void> f =
        hpx::parallel::uninitialized_fill_n(p,
            iterator(boost::begin(c)), c.size(),
            10);
    f.wait();

    std::size_t count = 0;
    std::for_each(boost::begin(c), boost::end(c),
        [&count](std::size_t v) -> void {
            HPX_TEST_EQ(v, std::size_t(10));
            ++count;
    });
    HPX_TEST_EQ(count, c.size());
}

template <typename IteratorTag>
void test_uninitialized_fill_n()
{
    using namespace hpx::parallel;
    test_uninitialized_fill_n(seq, IteratorTag());
    test_uninitialized_fill_n(par, IteratorTag());
    test_uninitialized_fill_n(par_vec, IteratorTag());

    test_uninitialized_fill_n_async(seq(task), IteratorTag());
    test_uninitialized_fill_n_async(par(task), IteratorTag());

#if defined(HPX_HAVE_GENERIC_EXECUTION_POLICY)
    test_uninitialized_fill_n(execution_policy(seq), IteratorTag());
    test_uninitialized_fill_n(execution_policy(par), IteratorTag());
    test_uninitialized_fill_n(execution_policy(par_vec), IteratorTag());

    test_uninitialized_fill_n(execution_policy(seq(task)), IteratorTag());
    test_uninitialized_fill_n(execution_policy(par(task)), IteratorTag());
#endif
}

void uninitialized_fill_n_test()
{
    test_uninitialized_fill_n<std::random_access_iterator_tag>();
    test_uninitialized_fill_n<std::forward_iterator_tag>();
}

///////////////////////////////////////////////////////////////////////////////
template <typename ExPolicy, typename IteratorTag>
void test_uninitialized_fill_n_exception(ExPolicy policy, IteratorTag)
{
    static_assert(
        hpx::parallel::is_execution_policy<ExPolicy>::value,
        "hpx::parallel::is_execution_policy<ExPolicy>::value");

    typedef std::vector<test::count_instances>::iterator base_iterator;
    typedef test::decorated_iterator <base_iterator, IteratorTag>
        decorated_iterator;

    std::vector<test::count_instances> c(10007);
    std::iota(boost::begin(c), boost::end(c), std::rand());

    boost::atomic<std::size_t> throw_after(std::rand() % c.size()); //-V104
    test::count_instances::instance_count.store(0);

    bool caught_exception = false;
    try {
        hpx::parallel::uninitialized_fill_n(policy,
            decorated_iterator(
                boost::begin(c),
                [&throw_after]()
                {
                    if (throw_after-- == 0)
                        throw std::runtime_error("test");
                }),
            c.size(),
            10);

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
    HPX_TEST_EQ(test::count_instances::instance_count.load(), std::size_t(0));
}

template <typename ExPolicy, typename IteratorTag>
void test_uninitialized_fill_n_exception_async(ExPolicy p, IteratorTag)
{
    typedef std::vector<test::count_instances>::iterator base_iterator;
    typedef test::decorated_iterator <base_iterator, IteratorTag>
        decorated_iterator;

    std::vector<test::count_instances> c(10007);
    std::iota(boost::begin(c), boost::end(c), std::rand());

    boost::atomic<std::size_t> throw_after(std::rand() % c.size()); //-V104
    test::count_instances::instance_count.store(0);

    bool caught_exception = false;
    bool returned_from_algorithm = false;
    try {
        hpx::future<void> f =
            hpx::parallel::uninitialized_fill_n(p,
                decorated_iterator(
                    boost::begin(c),
                    [&throw_after]()
                    {
                        if (throw_after-- == 0)
                            throw std::runtime_error("test");
                    }),
                c.size(),
                10);

        returned_from_algorithm = true;
        f.get();

        HPX_TEST(false);
    }
    catch (hpx::exception_list const& e) {
        caught_exception = true;
        test::test_num_exceptions<ExPolicy, IteratorTag>::call(p, e);
    }
    catch (...) {
        HPX_TEST(false);
    }

    HPX_TEST(caught_exception);
    HPX_TEST(returned_from_algorithm);
    HPX_TEST_EQ(test::count_instances::instance_count.load(), std::size_t(0));
}

template <typename IteratorTag>
void test_uninitialized_fill_n_exception()
{
    using namespace hpx::parallel;

    // If the execution policy object is of type vector_execution_policy,
    // std::terminate shall be called. therefore we do not test exceptions
    // with a vector execution policy
    test_uninitialized_fill_n_exception(seq, IteratorTag());
    test_uninitialized_fill_n_exception(par, IteratorTag());

    test_uninitialized_fill_n_exception_async(seq(task), IteratorTag());
    test_uninitialized_fill_n_exception_async(par(task), IteratorTag());

#if defined(HPX_HAVE_GENERIC_EXECUTION_POLICY)
    test_uninitialized_fill_n_exception(execution_policy(seq), IteratorTag());
    test_uninitialized_fill_n_exception(execution_policy(par), IteratorTag());

    test_uninitialized_fill_n_exception(execution_policy(seq(task)), IteratorTag());
    test_uninitialized_fill_n_exception(execution_policy(par(task)), IteratorTag());
#endif
}

void uninitialized_fill_n_exception_test()
{
    test_uninitialized_fill_n_exception<std::random_access_iterator_tag>();
    test_uninitialized_fill_n_exception<std::forward_iterator_tag>();
}

//////////////////////////////////////////////////////////////////////////////
template <typename ExPolicy, typename IteratorTag>
void test_uninitialized_fill_n_bad_alloc(ExPolicy policy, IteratorTag)
{
    static_assert(
        hpx::parallel::is_execution_policy<ExPolicy>::value,
        "hpx::parallel::is_execution_policy<ExPolicy>::value");

    typedef std::vector<test::count_instances>::iterator base_iterator;
    typedef test::decorated_iterator <base_iterator, IteratorTag>
        decorated_iterator;

    std::vector<test::count_instances> c(10007);
    std::iota(boost::begin(c), boost::end(c), std::rand());

    boost::atomic<std::size_t> throw_after(std::rand() % c.size()); //-V104
    test::count_instances::instance_count.store(0);

    bool caught_bad_alloc = false;
    try {
        hpx::parallel::uninitialized_fill_n(policy,
            decorated_iterator(
                boost::begin(c),
                [&throw_after]()
                {
                    if (throw_after-- == 0)
                        throw std::bad_alloc();
                }),
            c.size(),
            10);

        HPX_TEST(false);
    }
    catch (std::bad_alloc const&) {
        caught_bad_alloc = true;
    }
    catch (...) {
        HPX_TEST(false);
    }

    HPX_TEST(caught_bad_alloc);
    HPX_TEST_EQ(test::count_instances::instance_count.load(), std::size_t(0));
}

template <typename ExPolicy, typename IteratorTag>
void test_uninitialized_fill_n_bad_alloc_async(ExPolicy p, IteratorTag)
{
    typedef std::vector<test::count_instances>::iterator base_iterator;
    typedef test::decorated_iterator <base_iterator, IteratorTag>
        decorated_iterator;

    std::vector<test::count_instances> c(10007);
    std::iota(boost::begin(c), boost::end(c), std::rand());

    boost::atomic<std::size_t> throw_after(std::rand() % c.size()); //-V104
    test::count_instances::instance_count.store(0);

    bool caught_bad_alloc = false;
    bool returned_from_algorithm = false;
    try {
        hpx::future<void> f =
            hpx::parallel::uninitialized_fill_n(p,
                decorated_iterator(
                    boost::begin(c),
                    [&throw_after]()
                    {
                        if (throw_after-- == 0)
                            throw std::bad_alloc();
                    }),
                c.size(),
                10);

        returned_from_algorithm = true;
        f.get();

        HPX_TEST(false);
    }
    catch (std::bad_alloc const&) {
        caught_bad_alloc = true;
    }
    catch (...) {
        HPX_TEST(false);
    }

    HPX_TEST(caught_bad_alloc);
    HPX_TEST(returned_from_algorithm);
    HPX_TEST_EQ(test::count_instances::instance_count.load(), std::size_t(0));
}

template <typename IteratorTag>
void test_uninitialized_fill_n_bad_alloc()
{
    using namespace hpx::parallel;

    // If the execution policy object is of type vector_execution_policy,
    // std::terminate shall be called. therefore we do not test exceptions
    // with a vector execution policy
    test_uninitialized_fill_n_bad_alloc(seq, IteratorTag());
    test_uninitialized_fill_n_bad_alloc(par, IteratorTag());

    test_uninitialized_fill_n_bad_alloc_async(seq(task), IteratorTag());
    test_uninitialized_fill_n_bad_alloc_async(par(task), IteratorTag());

#if defined(HPX_HAVE_GENERIC_EXECUTION_POLICY)
    test_uninitialized_fill_n_bad_alloc(execution_policy(seq), IteratorTag());
    test_uninitialized_fill_n_bad_alloc(execution_policy(par), IteratorTag());

    test_uninitialized_fill_n_bad_alloc(execution_policy(seq(task)), IteratorTag());
    test_uninitialized_fill_n_bad_alloc(execution_policy(par(task)), IteratorTag());
#endif
}

void uninitialized_fill_n_bad_alloc_test()
{
    test_uninitialized_fill_n_bad_alloc<std::random_access_iterator_tag>();
    test_uninitialized_fill_n_bad_alloc<std::forward_iterator_tag>();
}

int hpx_main(boost::program_options::variables_map& vm)
{
    unsigned int seed = (unsigned int)std::time(0);
    if (vm.count("seed"))
        seed = vm["seed"].as<unsigned int>();

    std::cout << "using seed: " << seed << std::endl;
    std::srand(seed);

    uninitialized_fill_n_test();
    uninitialized_fill_n_exception_test();
    uninitialized_fill_n_bad_alloc_test();
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
    std::vector<std::string> cfg;
    cfg.push_back("hpx.os_threads=" +
        std::to_string(hpx::threads::hardware_concurrency()));

    // Initialize and run HPX
    HPX_TEST_EQ_MSG(hpx::init(desc_commandline, argc, argv, cfg), 0,
        "HPX main exited with non-zero status");

    return hpx::util::report_errors();
}
