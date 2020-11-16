//  Copyright (c) 2016-2020 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// This is more of an integration test than a unit test. The goal is to verify
// that the resource manager used to assign processing units to the executors
// does the right thing.

#include <hpx/config.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/include/parallel_execution.hpp>
#include <hpx/include/parallel_executors.hpp>
#include <hpx/modules/testing.hpp>
#include <hpx/thread_executors/resource_manager.hpp>

#include <atomic>
#include <cstddef>
#include <string>
#include <type_traits>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
namespace test {
    struct dummy_parameters
    {
        dummy_parameters() = default;
    };

    static constexpr dummy_parameters dummy{};
}    // namespace test

namespace hpx { namespace parallel { namespace execution {
    template <>
    struct is_executor_parameters<test::dummy_parameters> : std::true_type
    {
    };
}}}    // namespace hpx::parallel::execution

///////////////////////////////////////////////////////////////////////////////
void verify_resource_allocation(std::size_t num_execs, std::size_t num_pus)
{
    std::vector<hpx::threads::resource_allocation> alloc_data =
        hpx::threads::get_resource_allocation();

    HPX_TEST_EQ(num_execs, alloc_data.size());
    for (hpx::threads::resource_allocation const& data : alloc_data)
    {
        HPX_TEST_EQ(num_pus, data.core_ids_.size());
    }
}

///////////////////////////////////////////////////////////////////////////////
template <typename Executor>
void test_executors(std::size_t processing_units, std::size_t num_pus)
{
    std::atomic<std::size_t> count_invocations(0);
    std::size_t const num_tasks = 100;

    std::size_t num_execs = processing_units / num_pus;

    {
        // create as many executors as we have cores available using num_pus
        // processing unit each
        std::vector<Executor> execs;
        for (std::size_t i = 0; i != num_execs; ++i)
            execs.push_back(Executor(num_pus));

        verify_resource_allocation(num_execs, num_pus);

        // give executors a chance to get started
        hpx::this_thread::yield();

        for (Executor& exec : execs)
        {
            HPX_TEST_EQ(hpx::parallel::execution::processing_units_count(
                            test::dummy, exec),
                num_pus);
        }

        // schedule a couple of tasks on each of the executors
        for (Executor& exec : execs)
        {
            for (int i = 0; i != num_tasks; ++i)
            {
                hpx::parallel::execution::post(
                    exec, [&count_invocations]() { ++count_invocations; });
            }
        }

        // test again
        for (Executor& exec : execs)
        {
            HPX_TEST_EQ(hpx::parallel::execution::processing_units_count(
                            test::dummy, exec),
                num_pus);
        }
    }

    HPX_TEST_EQ(count_invocations, num_execs * num_tasks);
}

void test_executors(std::size_t num_pus)
{
    using namespace hpx::parallel::execution;
    std::size_t processing_units = hpx::get_os_thread_count();

    processing_units = (processing_units / num_pus) * num_pus;

#if defined(HPX_HAVE_LOCAL_SCHEDULER)
    test_executors<local_queue_executor>(processing_units, num_pus);
#endif
#if defined(HPX_HAVE_STATIC_SCHEDULER)
    test_executors<static_queue_executor>(processing_units, num_pus);
#endif
    test_executors<local_priority_queue_executor>(processing_units, num_pus);
#if defined(HPX_HAVE_STATIC_PRIORITY_SCHEDULER)
    test_executors<static_priority_queue_executor>(processing_units, num_pus);
#endif
}

///////////////////////////////////////////////////////////////////////////////
template <typename Executor>
void test_executors_shrink(std::size_t processing_units, std::size_t num_pus)
{
    std::atomic<std::size_t> count_invocations(0);
    std::size_t const num_tasks = 100;

    // create one executor which can give back processing units
    Executor shrink_exec(processing_units);

    // give the executor a chance to get started
    hpx::this_thread::yield();

    HPX_TEST_EQ(hpx::parallel::execution::processing_units_count(
                    test::dummy, shrink_exec),
        processing_units);

    std::size_t num_execs = (processing_units - 1) / num_pus;

    {
        // create as many executors as we have cores available using num_pus
        // processing unit each
        std::vector<Executor> execs;
        for (std::size_t i = 0; i != num_execs; ++i)
            execs.push_back(Executor(num_pus));

        // give executors a chance to get started
        hpx::this_thread::yield();

        for (Executor& exec : execs)
        {
            HPX_TEST_EQ(hpx::parallel::execution::processing_units_count(
                            test::dummy, exec),
                num_pus);
        }

        // the main executor should run on a reduced amount of cores
        HPX_TEST_EQ(hpx::parallel::execution::processing_units_count(
                        test::dummy, shrink_exec),
            processing_units - num_execs * num_pus);

        // schedule a couple of tasks on each of the executors
        for (Executor& exec : execs)
        {
            for (int i = 0; i != num_tasks; ++i)
            {
                hpx::parallel::execution::post(
                    exec, [&count_invocations]() { ++count_invocations; });
            }
        }

        // test again
        for (Executor& exec : execs)
        {
            HPX_TEST_EQ(hpx::parallel::execution::processing_units_count(
                            test::dummy, exec),
                num_pus);
        }

        // the main executor should run on a reduced amount of cores
        HPX_TEST_EQ(hpx::parallel::execution::processing_units_count(
                        test::dummy, shrink_exec),
            processing_units - num_execs * num_pus);
    }

    HPX_TEST_EQ(count_invocations, num_execs * num_tasks);
}

void test_executors_shrink(std::size_t num_pus)
{
    using namespace hpx::parallel::execution;
    std::size_t processing_units = hpx::get_os_thread_count();

    // we should have at least one processing unit
    processing_units =
        (std::max)((processing_units / num_pus) * num_pus, std::size_t(1));

#if defined(HPX_HAVE_LOCAL_SCHEDULER)
    test_executors_shrink<local_queue_executor>(processing_units, num_pus);
#endif
#if defined(HPX_HAVE_STATIC_SCHEDULER)
    test_executors_shrink<static_queue_executor>(processing_units, num_pus);
#endif
    test_executors_shrink<local_priority_queue_executor>(
        processing_units, num_pus);
#if defined(HPX_HAVE_STATIC_PRIORITY_SCHEDULER)
    test_executors_shrink<static_priority_queue_executor>(
        processing_units, num_pus);
#endif
}

///////////////////////////////////////////////////////////////////////////////
int hpx_main()
{
    // test without over-subscription
    test_executors(1);
    test_executors(2);
    test_executors(4);

    // test over-subscription, where schedulers can be forced to shrink
    test_executors_shrink(1);
    test_executors_shrink(2);
    test_executors_shrink(4);

    return hpx::finalize();
}

///////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
    // By default this test should run on all available cores
    std::vector<std::string> cfg;
    cfg.push_back("hpx.os_threads=all");

    hpx::init_params init_args;
    init_args.cfg = cfg;

    HPX_TEST_EQ_MSG(hpx::init(argc, argv, init_args), 0,
        "HPX main exited with non-zero status");

    return hpx::util::report_errors();
}
