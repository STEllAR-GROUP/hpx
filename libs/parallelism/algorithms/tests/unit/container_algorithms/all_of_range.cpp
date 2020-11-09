//  Copyright (c) 2014-2020 Hartmut Kaiser
//                2017 Bruno Pitrus

//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/include/parallel_all_any_none_of.hpp>
#include <hpx/modules/testing.hpp>
#include <hpx/parallel/util/projection_identity.hpp>

#include <cstddef>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>

#include "test_utils.hpp"

///////////////////////////////////////////////////////////////////////////////
template <typename IteratorTag,
    typename Proj = hpx::parallel::util::projection_identity>
void test_all_of_seq(IteratorTag, Proj proj = Proj())
{
    std::size_t iseq[] = {0, 23, 10007};
    for (std::size_t i : iseq)
    {
        std::vector<std::size_t> c =
            test::fill_all_any_none(10007, i);    //-V106

        bool result = hpx::ranges::all_of(
            c, [](std::size_t v) { return v != 0; }, proj);

        // verify values
        bool expected = std::all_of(std::begin(c), std::end(c),
            [proj](std::size_t v) { return proj(v) != 0; });

        HPX_TEST_EQ(result, expected);
    }
}

template <typename ExPolicy, typename IteratorTag,
    typename Proj = hpx::parallel::util::projection_identity>
void test_all_of(ExPolicy&& policy, IteratorTag, Proj proj = Proj())
{
    static_assert(hpx::is_execution_policy<ExPolicy>::value,
        "hpx::is_execution_policy<ExPolicy>::value");

    std::size_t iseq[] = {0, 23, 10007};
    for (std::size_t i : iseq)
    {
        std::vector<std::size_t> c =
            test::fill_all_any_none(10007, i);    //-V106

        bool result = hpx::ranges::all_of(
            policy, c, [](std::size_t v) { return v != 0; }, proj);

        // verify values
        bool expected = std::all_of(std::begin(c), std::end(c),
            [proj](std::size_t v) { return proj(v) != 0; });

        HPX_TEST_EQ(result, expected);
    }
}

template <typename ExPolicy, typename IteratorTag,
    typename Proj = hpx::parallel::util::projection_identity>
void test_all_of_async(ExPolicy p, IteratorTag, Proj proj = Proj())
{
    std::size_t iseq[] = {0, 23, 10007};
    for (std::size_t i : iseq)
    {
        std::vector<std::size_t> c =
            test::fill_all_any_none(10007, i);    //-V106

        hpx::future<bool> f = hpx::ranges::all_of(
            p, c, [](std::size_t v) { return v != 0; }, proj);
        f.wait();

        // verify values
        bool expected = std::all_of(std::begin(c), std::end(c),
            [proj](std::size_t v) { return proj(v) != 0; });

        HPX_TEST_EQ(expected, f.get());
    }
}

template <typename IteratorTag>
void test_all_of()
{
    struct proj
    {
        //This projection should cause tests to fail if it is not applied
        //because it causes predicate to evaluate the opposite
        constexpr std::size_t operator()(std::size_t x) const
        {
            return !static_cast<bool>(x);
        }
    };
    using namespace hpx::execution;

    test_all_of_seq(IteratorTag());
    test_all_of_seq(IteratorTag(), proj());

    test_all_of(seq, IteratorTag());
    test_all_of(par, IteratorTag());
    test_all_of(par_unseq, IteratorTag());

    test_all_of(seq, IteratorTag(), proj());
    test_all_of(par, IteratorTag(), proj());
    test_all_of(par_unseq, IteratorTag(), proj());

    test_all_of_async(seq(task), IteratorTag());
    test_all_of_async(par(task), IteratorTag());

    test_all_of_async(seq(task), IteratorTag(), proj());
    test_all_of_async(par(task), IteratorTag(), proj());
}

// template <typename IteratorTag>
// void test_all_of_exec()
// {
//     using namespace hpx::execution;
//
//     {
//         hpx::threads::executors::local_priority_queue_executor exec;
//         test_all_of(par(exec), IteratorTag());
//     }
//     {
//         hpx::threads::executors::local_priority_queue_executor exec;
//         test_all_of(task(exec), IteratorTag());
//     }
//
//     {
//         hpx::threads::executors::local_priority_queue_executor exec;
//         test_all_of(execution_policy(par(exec)), IteratorTag());
//     }
//     {
//         hpx::threads::executors::local_priority_queue_executor exec;
//         test_all_of(execution_policy(task(exec)), IteratorTag());
//     }
// }

void all_of_test()
{
    test_all_of<std::random_access_iterator_tag>();
    test_all_of<std::forward_iterator_tag>();
}

///////////////////////////////////////////////////////////////////////////////
template <typename IteratorTag>
void test_all_of_exception(IteratorTag)
{
    std::size_t iseq[] = {0, 23, 10007};
    for (std::size_t i : iseq)
    {
        std::vector<std::size_t> c =
            test::fill_all_any_none(10007, i);    //-V106

        bool caught_exception = false;
        try
        {
            hpx::ranges::all_of(c, [](std::size_t v) {
                return throw std::runtime_error("test"), v != 0;
            });

            HPX_TEST(false);
        }
        catch (hpx::exception_list const& e)
        {
            caught_exception = true;
            test::test_num_exceptions<hpx::execution::sequenced_policy,
                IteratorTag>::call(hpx::execution::seq, e);
        }
        catch (...)
        {
            HPX_TEST(false);
        }

        HPX_TEST(caught_exception);
    }
}

template <typename ExPolicy, typename IteratorTag>
void test_all_of_exception(ExPolicy&& policy, IteratorTag)
{
    static_assert(hpx::is_execution_policy<ExPolicy>::value,
        "hpx::is_execution_policy<ExPolicy>::value");

    std::size_t iseq[] = {0, 23, 10007};
    for (std::size_t i : iseq)
    {
        std::vector<std::size_t> c =
            test::fill_all_any_none(10007, i);    //-V106

        bool caught_exception = false;
        try
        {
            hpx::ranges::all_of(policy, c, [](std::size_t v) {
                return throw std::runtime_error("test"), v != 0;
            });

            HPX_TEST(false);
        }
        catch (hpx::exception_list const& e)
        {
            caught_exception = true;
            test::test_num_exceptions<ExPolicy, IteratorTag>::call(policy, e);
        }
        catch (...)
        {
            HPX_TEST(false);
        }

        HPX_TEST(caught_exception);
    }
}

template <typename ExPolicy, typename IteratorTag>
void test_all_of_exception_async(ExPolicy p, IteratorTag)
{
    std::size_t iseq[] = {0, 23, 10007};
    for (std::size_t i : iseq)
    {
        std::vector<std::size_t> c =
            test::fill_all_any_none(10007, i);    //-V106

        bool caught_exception = false;
        bool returned_from_algorithm = false;
        try
        {
            hpx::future<void> f = hpx::ranges::all_of(p, c, [](std::size_t v) {
                return throw std::runtime_error("test"), v != 0;
            });
            returned_from_algorithm = true;
            f.get();

            HPX_TEST(false);
        }
        catch (hpx::exception_list const& e)
        {
            caught_exception = true;
            test::test_num_exceptions<ExPolicy, IteratorTag>::call(p, e);
        }
        catch (...)
        {
            HPX_TEST(false);
        }

        HPX_TEST(caught_exception);
        HPX_TEST(returned_from_algorithm);
    }
}

template <typename IteratorTag>
void test_all_of_exception()
{
    using namespace hpx::execution;

    test_all_of_exception(IteratorTag());

    // If the execution policy object is of type vector_execution_policy,
    // std::terminate shall be called. therefore we do not test exceptions
    // with a vector execution policy
    test_all_of_exception(seq, IteratorTag());
    test_all_of_exception(par, IteratorTag());

    test_all_of_exception_async(seq(task), IteratorTag());
    test_all_of_exception_async(par(task), IteratorTag());
}

void all_of_exception_test()
{
    test_all_of_exception<std::random_access_iterator_tag>();
    test_all_of_exception<std::forward_iterator_tag>();
}

///////////////////////////////////////////////////////////////////////////////
template <typename ExPolicy, typename IteratorTag>
void test_all_of_bad_alloc(ExPolicy&& policy, IteratorTag)
{
    static_assert(hpx::is_execution_policy<ExPolicy>::value,
        "hpx::is_execution_policy<ExPolicy>::value");

    std::size_t iseq[] = {0, 23, 10007};
    for (std::size_t i : iseq)
    {
        std::vector<std::size_t> c =
            test::fill_all_any_none(10007, i);    //-V106

        bool caught_exception = false;
        try
        {
            hpx::ranges::all_of(policy, c,
                [](std::size_t v) { return throw std::bad_alloc(), v != 0; });

            HPX_TEST(false);
        }
        catch (std::bad_alloc const&)
        {
            caught_exception = true;
        }
        catch (...)
        {
            HPX_TEST(false);
        }

        HPX_TEST(caught_exception);
    }
}

template <typename ExPolicy, typename IteratorTag>
void test_all_of_bad_alloc_async(ExPolicy p, IteratorTag)
{
    std::size_t iseq[] = {0, 23, 10007};
    for (std::size_t i : iseq)
    {
        std::vector<std::size_t> c =
            test::fill_all_any_none(10007, i);    //-V106

        bool caught_exception = false;
        bool returned_from_algorithm = false;
        try
        {
            hpx::future<void> f = hpx::ranges::all_of(p, c,
                [](std::size_t v) { return throw std::bad_alloc(), v != 0; });
            returned_from_algorithm = true;
            f.get();

            HPX_TEST(false);
        }
        catch (std::bad_alloc const&)
        {
            caught_exception = true;
        }
        catch (...)
        {
            HPX_TEST(false);
        }

        HPX_TEST(caught_exception);
        HPX_TEST(returned_from_algorithm);
    }
}

template <typename IteratorTag>
void test_all_of_bad_alloc()
{
    using namespace hpx::execution;

    // If the execution policy object is of type vector_execution_policy,
    // std::terminate shall be called. therefore we do not test exceptions
    // with a vector execution policy
    test_all_of_bad_alloc(seq, IteratorTag());
    test_all_of_bad_alloc(par, IteratorTag());

    test_all_of_bad_alloc_async(seq(task), IteratorTag());
    test_all_of_bad_alloc_async(par(task), IteratorTag());
}

void all_of_bad_alloc_test()
{
    test_all_of_bad_alloc<std::random_access_iterator_tag>();
    test_all_of_bad_alloc<std::forward_iterator_tag>();
}

///////////////////////////////////////////////////////////////////////////////
int hpx_main(hpx::program_options::variables_map& vm)
{
    unsigned int seed = (unsigned int) std::time(nullptr);
    if (vm.count("seed"))
        seed = vm["seed"].as<unsigned int>();

    std::cout << "using seed: " << seed << std::endl;
    std::srand(seed);

    all_of_test();
    all_of_exception_test();
    all_of_bad_alloc_test();
    return hpx::finalize();
}

int main(int argc, char* argv[])
{
    // add command line option which controls the random number generator seed
    using namespace hpx::program_options;
    options_description desc_commandline(
        "Usage: " HPX_APPLICATION_STRING " [options]");

    desc_commandline.add_options()("seed,s", value<unsigned int>(),
        "the random number generator seed to use for this run");

    // By default this test should run on all available cores
    std::vector<std::string> const cfg = {"hpx.os_threads=all"};

    // Initialize and run HPX
    hpx::init_params init_args;
    init_args.desc_cmdline = desc_commandline;
    init_args.cfg = cfg;

    HPX_TEST_EQ_MSG(hpx::init(argc, argv, init_args), 0,
        "HPX main exited with non-zero status");

    return hpx::util::report_errors();
}
