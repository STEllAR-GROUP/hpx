//  Copyright (c) 2014 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>
#include <hpx/hpx.hpp>
#include <hpx/include/parallel_all_any_none_of.hpp>
#include <hpx/util/lightweight_test.hpp>

#include <boost/range/functions.hpp>

#include "test_utils.hpp"

///////////////////////////////////////////////////////////////////////////////
template <typename ExPolicy, typename IteratorTag>
void test_any_of(ExPolicy policy, IteratorTag)
{
    static_assert(
        hpx::parallel::is_execution_policy<ExPolicy>::value,
        "hpx::parallel::is_execution_policy<ExPolicy>::value");

    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::size_t iseq[] = { 0, 23, 10007 };
    for (std::size_t i: iseq)
    {
        std::vector<std::size_t> c = test::fill_all_any_none(10007, i);

        bool result =
            hpx::parallel::any_of(policy,
                iterator(boost::begin(c)), iterator(boost::end(c)),
                [](std::size_t v) {
                    return v != 0;
                });

        // verify values
        bool expected =
            std::any_of(boost::begin(c), boost::end(c),
                [](std::size_t v) {
                    return v != 0;
                });

        HPX_TEST_EQ(result, expected);
    }
}

template <typename ExPolicy, typename IteratorTag>
void test_any_of_async(ExPolicy p, IteratorTag)
{
    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::size_t iseq[] = { 0, 23, 10007 };
    for (std::size_t i: iseq)
    {
        std::vector<std::size_t> c = test::fill_all_any_none(10007, i);

        hpx::future<bool> f =
            hpx::parallel::any_of(p,
                iterator(boost::begin(c)), iterator(boost::end(c)),
                [](std::size_t v) {
                    return v != 0;
                });
        f.wait();

        // verify values
        bool expected =
            std::any_of(boost::begin(c), boost::end(c),
                [](std::size_t v) {
                    return v != 0;
                });

        HPX_TEST_EQ(expected, f.get());
    }
}

template <typename IteratorTag>
void test_any_of()
{
    using namespace hpx::parallel;

    test_any_of(seq, IteratorTag());
    test_any_of(par, IteratorTag());
    test_any_of(par_vec, IteratorTag());

    test_any_of_async(seq(task), IteratorTag());
    test_any_of_async(par(task), IteratorTag());

    test_any_of(execution_policy(seq), IteratorTag());
    test_any_of(execution_policy(par), IteratorTag());
    test_any_of(execution_policy(par_vec), IteratorTag());

    test_any_of(execution_policy(seq(task)), IteratorTag());
    test_any_of(execution_policy(par(task)), IteratorTag());
}

// template <typename IteratorTag>
// void test_any_of_exec()
// {
//     using namespace hpx::parallel;
//
//     {
//         hpx::threads::executors::local_priority_queue_executor exec;
//         test_any_of(par(exec), IteratorTag());
//     }
//     {
//         hpx::threads::executors::local_priority_queue_executor exec;
//         test_any_of(task(exec), IteratorTag());
//     }
//
//     {
//         hpx::threads::executors::local_priority_queue_executor exec;
//         test_any_of(execution_policy(par(exec)), IteratorTag());
//     }
//     {
//         hpx::threads::executors::local_priority_queue_executor exec;
//         test_any_of(execution_policy(task(exec)), IteratorTag());
//     }
// }

void any_of_test()
{
    test_any_of<std::random_access_iterator_tag>();
    test_any_of<std::forward_iterator_tag>();
    test_any_of<std::input_iterator_tag>();

//     test_any_of_exec<std::random_access_iterator_tag>();
//     test_any_of_exec<std::forward_iterator_tag>();
//     test_any_of_exec<std::input_iterator_tag>();
}

///////////////////////////////////////////////////////////////////////////////
template <typename ExPolicy, typename IteratorTag>
void test_any_of_exception(ExPolicy policy, IteratorTag)
{
    static_assert(
        hpx::parallel::is_execution_policy<ExPolicy>::value,
        "hpx::parallel::is_execution_policy<ExPolicy>::value");

    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::size_t iseq[] = { 0, 23, 10007 };
    for (std::size_t i: iseq)
    {
        std::vector<std::size_t> c = test::fill_all_any_none(10007, i);

        bool caught_exception = false;
        try {
            hpx::parallel::any_of(policy,
                iterator(boost::begin(c)), iterator(boost::end(c)),
                [](std::size_t v) {
                    return throw std::runtime_error("test"), v != 0;
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
}

template <typename ExPolicy, typename IteratorTag>
void test_any_of_exception_async(ExPolicy p, IteratorTag)
{
    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::size_t iseq[] = { 0, 23, 10007 };
    for (std::size_t i : iseq)
    {
        std::vector<std::size_t> c = test::fill_all_any_none(10007, i);

        bool caught_exception = false;
        bool returned_from_algorithm = false;
        try {
            hpx::future<void> f =
                hpx::parallel::any_of(p,
                    iterator(boost::begin(c)), iterator(boost::end(c)),
                    [](std::size_t v) {
                        return throw std::runtime_error("test"), v != 0;
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
}

template <typename IteratorTag>
void test_any_of_exception()
{
    using namespace hpx::parallel;

    // If the execution policy object is of type vector_execution_policy,
    // std::terminate shall be called. therefore we do not test exceptions
    // with a vector execution policy
    test_any_of_exception(seq, IteratorTag());
    test_any_of_exception(par, IteratorTag());

    test_any_of_exception_async(seq(task), IteratorTag());
    test_any_of_exception_async(par(task), IteratorTag());

    test_any_of_exception(execution_policy(seq), IteratorTag());
    test_any_of_exception(execution_policy(par), IteratorTag());

    test_any_of_exception(execution_policy(seq(task)), IteratorTag());
    test_any_of_exception(execution_policy(par(task)), IteratorTag());
}

void any_of_exception_test()
{
    test_any_of_exception<std::random_access_iterator_tag>();
    test_any_of_exception<std::forward_iterator_tag>();
    test_any_of_exception<std::input_iterator_tag>();
}

///////////////////////////////////////////////////////////////////////////////
template <typename ExPolicy, typename IteratorTag>
void test_any_of_bad_alloc(ExPolicy policy, IteratorTag)
{
    static_assert(
        hpx::parallel::is_execution_policy<ExPolicy>::value,
        "hpx::parallel::is_execution_policy<ExPolicy>::value");

    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::size_t iseq[] = { 0, 23, 10007 };
    for (std::size_t i : iseq)
    {
        std::vector<std::size_t> c = test::fill_all_any_none(10007, i);

        bool caught_exception = false;
        try {
            hpx::parallel::any_of(policy,
                iterator(boost::begin(c)), iterator(boost::end(c)),
                [](std::size_t v) {
                    return throw std::bad_alloc(), v != 0;
                });

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
}

template <typename ExPolicy, typename IteratorTag>
void test_any_of_bad_alloc_async(ExPolicy p, IteratorTag)
{
    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::size_t iseq[] = { 0, 23, 10007 };
    for (std::size_t i : iseq)
    {
        std::vector<std::size_t> c = test::fill_all_any_none(10007, i);

        bool caught_exception = false;
        bool returned_from_algorithm = false;
        try {
            hpx::future<void> f =
                hpx::parallel::any_of(p,
                    iterator(boost::begin(c)), iterator(boost::end(c)),
                    [](std::size_t v) {
                        return throw std::bad_alloc(), v != 0;
                    });
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
}

template <typename IteratorTag>
void test_any_of_bad_alloc()
{
    using namespace hpx::parallel;

    // If the execution policy object is of type vector_execution_policy,
    // std::terminate shall be called. therefore we do not test exceptions
    // with a vector execution policy
    test_any_of_bad_alloc(seq, IteratorTag());
    test_any_of_bad_alloc(par, IteratorTag());

    test_any_of_bad_alloc_async(seq(task), IteratorTag());
    test_any_of_bad_alloc_async(par(task), IteratorTag());

    test_any_of_bad_alloc(execution_policy(seq), IteratorTag());
    test_any_of_bad_alloc(execution_policy(par), IteratorTag());

    test_any_of_bad_alloc(execution_policy(seq(task)), IteratorTag());
    test_any_of_bad_alloc(execution_policy(par(task)), IteratorTag());
}

void any_of_bad_alloc_test()
{
    test_any_of_bad_alloc<std::random_access_iterator_tag>();
    test_any_of_bad_alloc<std::forward_iterator_tag>();
    test_any_of_bad_alloc<std::input_iterator_tag>();
}

///////////////////////////////////////////////////////////////////////////////
int hpx_main(boost::program_options::variables_map& vm)
{
    unsigned int seed = (unsigned int)std::time(0);
    if (vm.count("seed"))
        seed = vm["seed"].as<unsigned int>();

    std::cout << "using seed: " << seed << std::endl;
    std::srand(seed);

    any_of_test();
    any_of_exception_test();
    any_of_bad_alloc_test();
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


