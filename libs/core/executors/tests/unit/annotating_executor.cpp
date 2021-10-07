//  Copyright (c) 2007-2021 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/local/config.hpp>

#if defined(HPX_HAVE_THREAD_DESCRIPTION)
#include <hpx/local/execution.hpp>
#include <hpx/local/future.hpp>
#include <hpx/local/init.hpp>
#include <hpx/local/latch.hpp>
#include <hpx/local/thread.hpp>
#include <hpx/modules/properties.hpp>
#include <hpx/modules/testing.hpp>

#include <cstdlib>
#include <functional>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
std::string annotation;

void test_post_f(int passed_through, hpx::lcos::local::latch& l)
{
    HPX_TEST_EQ(passed_through, 42);

    annotation =
        hpx::threads::get_thread_description(hpx::threads::get_self_id())
            .get_description();

    l.count_down(1);
}

template <typename Executor>
void test_post(Executor&& executor)
{
    std::string desc("test_post");
    {
        auto exec = hpx::experimental::prefer(
            hpx::execution::experimental::with_annotation, executor, desc);

        hpx::lcos::local::latch l(2);
        hpx::parallel::execution::post(exec, &test_post_f, 42, std::ref(l));
        l.arrive_and_wait();

        HPX_TEST_EQ(annotation, desc);
        HPX_TEST_EQ(annotation,
            std::string(hpx::execution::experimental::get_annotation(exec)));
    }

    {
        annotation.clear();
        auto exec =
            hpx::execution::experimental::with_annotation(executor, desc);

        hpx::lcos::local::latch l(2);
        hpx::parallel::execution::post(exec, &test_post_f, 42, std::ref(l));
        l.arrive_and_wait();

        HPX_TEST_EQ(annotation, desc);
        HPX_TEST_EQ(annotation,
            std::string(hpx::execution::experimental::get_annotation(exec)));
    }
}

///////////////////////////////////////////////////////////////////////////////
void test(int passed_through)
{
    HPX_TEST_EQ(passed_through, 42);

    annotation =
        hpx::threads::get_thread_description(hpx::threads::get_self_id())
            .get_description();
}

template <typename Executor>
void test_sync(Executor&& executor)
{
    std::string desc("test_sync");
    {
        auto exec = hpx::experimental::prefer(
            hpx::execution::experimental::with_annotation, executor, desc);

        hpx::parallel::execution::sync_execute(exec, &test, 42);

        HPX_TEST_EQ(annotation, desc);
        HPX_TEST_EQ(annotation,
            std::string(hpx::execution::experimental::get_annotation(exec)));
    }

    {
        annotation.clear();
        auto exec =
            hpx::execution::experimental::with_annotation(executor, desc);

        hpx::parallel::execution::sync_execute(exec, &test, 42);

        HPX_TEST_EQ(annotation, desc);
        HPX_TEST_EQ(annotation,
            std::string(hpx::execution::experimental::get_annotation(exec)));
    }
}

template <typename Executor>
void test_async(Executor&& executor)
{
    std::string desc("test_async");
    {
        auto exec = hpx::experimental::prefer(
            hpx::execution::experimental::with_annotation, executor, desc);

        hpx::parallel::execution::async_execute(exec, &test, 42).get();

        HPX_TEST_EQ(annotation, desc);
        HPX_TEST_EQ(annotation,
            std::string(hpx::execution::experimental::get_annotation(exec)));
    }

    {
        annotation.clear();
        auto exec =
            hpx::execution::experimental::with_annotation(executor, desc);

        hpx::parallel::execution::async_execute(exec, &test, 42).get();

        HPX_TEST_EQ(annotation, desc);
        HPX_TEST_EQ(annotation,
            std::string(hpx::execution::experimental::get_annotation(exec)));
    }
}

///////////////////////////////////////////////////////////////////////////////
void test_f(hpx::future<void> f, int passed_through)
{
    HPX_TEST(f.is_ready());    // make sure, future is ready

    f.get();    // propagate exceptions

    HPX_TEST_EQ(passed_through, 42);

    annotation =
        hpx::threads::get_thread_description(hpx::threads::get_self_id())
            .get_description();
}

template <typename Executor>
void test_then(Executor&& executor)
{
    hpx::future<void> f = hpx::make_ready_future();

    std::string desc("test_then");
    {
        auto exec = hpx::experimental::prefer(
            hpx::execution::experimental::with_annotation, executor, desc);

        hpx::parallel::execution::then_execute(exec, &test_f, f, 42).get();

        HPX_TEST_EQ(annotation, desc);
        HPX_TEST_EQ(annotation,
            std::string(hpx::execution::experimental::get_annotation(exec)));
    }

    {
        annotation.clear();
        auto exec =
            hpx::execution::experimental::with_annotation(executor, desc);

        hpx::parallel::execution::then_execute(exec, &test_f, f, 42).get();

        HPX_TEST_EQ(annotation, desc);
        HPX_TEST_EQ(annotation,
            std::string(hpx::execution::experimental::get_annotation(exec)));
    }
}

///////////////////////////////////////////////////////////////////////////////
void bulk_test(int seq, int passed_through)    //-V813
{
    HPX_TEST_EQ(passed_through, 42);

    if (seq == 0)
    {
        annotation =
            hpx::threads::get_thread_description(hpx::threads::get_self_id())
                .get_description();
    }
}

template <typename Executor>
void test_bulk_sync(Executor&& executor)
{
    using hpx::util::placeholders::_1;
    using hpx::util::placeholders::_2;

    std::string desc("test_bulk_sync");
    {
        auto exec = hpx::experimental::prefer(
            hpx::execution::experimental::with_annotation, executor, desc);

        hpx::parallel::execution::bulk_sync_execute(
            exec, hpx::util::bind(&bulk_test, _1, _2), 107, 42);

        HPX_TEST_EQ(annotation, desc);
        HPX_TEST_EQ(annotation,
            std::string(hpx::execution::experimental::get_annotation(exec)));

        annotation.clear();
        hpx::parallel::execution::bulk_sync_execute(exec, &bulk_test, 107, 42);

        HPX_TEST_EQ(annotation, desc);
        HPX_TEST_EQ(annotation,
            std::string(hpx::execution::experimental::get_annotation(exec)));
    }

    {
        auto exec =
            hpx::execution::experimental::with_annotation(executor, desc);

        annotation.clear();
        hpx::parallel::execution::bulk_sync_execute(
            exec, hpx::util::bind(&bulk_test, _1, _2), 107, 42);

        HPX_TEST_EQ(annotation, desc);
        HPX_TEST_EQ(annotation,
            std::string(hpx::execution::experimental::get_annotation(exec)));

        annotation.clear();
        hpx::parallel::execution::bulk_sync_execute(exec, &bulk_test, 107, 42);

        HPX_TEST_EQ(annotation, desc);
        HPX_TEST_EQ(annotation,
            std::string(hpx::execution::experimental::get_annotation(exec)));
    }
}

///////////////////////////////////////////////////////////////////////////////
template <typename Executor>
void test_bulk_async(Executor&& executor)
{
    using hpx::util::placeholders::_1;
    using hpx::util::placeholders::_2;

    std::string desc("test_bulk_async");
    {
        auto exec = hpx::experimental::prefer(
            hpx::execution::experimental::with_annotation, executor, desc);

        hpx::when_all(hpx::parallel::execution::bulk_async_execute(
                          exec, hpx::util::bind(&bulk_test, _1, _2), 107, 42))
            .get();

        HPX_TEST_EQ(annotation, desc);
        HPX_TEST_EQ(annotation,
            std::string(hpx::execution::experimental::get_annotation(exec)));

        annotation.clear();
        hpx::when_all(hpx::parallel::execution::bulk_async_execute(
                          exec, &bulk_test, 107, 42))
            .get();

        HPX_TEST_EQ(annotation, desc);
        HPX_TEST_EQ(annotation,
            std::string(hpx::execution::experimental::get_annotation(exec)));
    }

    {
        auto exec =
            hpx::execution::experimental::with_annotation(executor, desc);

        annotation.clear();
        hpx::when_all(hpx::parallel::execution::bulk_async_execute(
                          exec, hpx::util::bind(&bulk_test, _1, _2), 107, 42))
            .get();

        HPX_TEST_EQ(annotation, desc);
        HPX_TEST_EQ(annotation,
            std::string(hpx::execution::experimental::get_annotation(exec)));

        annotation.clear();
        hpx::when_all(hpx::parallel::execution::bulk_async_execute(
                          exec, &bulk_test, 107, 42))
            .get();

        HPX_TEST_EQ(annotation, desc);
        HPX_TEST_EQ(annotation,
            std::string(hpx::execution::experimental::get_annotation(exec)));
    }
}

///////////////////////////////////////////////////////////////////////////////
void bulk_test_f(int seq, hpx::shared_future<void> f,
    int passed_through)    //-V813
{
    HPX_TEST(f.is_ready());    // make sure, future is ready

    f.get();    // propagate exceptions

    HPX_TEST_EQ(passed_through, 42);

    if (seq == 0)
    {
        annotation =
            hpx::threads::get_thread_description(hpx::threads::get_self_id())
                .get_description();
    }
}

template <typename Executor>
void test_bulk_then(Executor&& executor)
{
    using hpx::util::placeholders::_1;
    using hpx::util::placeholders::_2;
    using hpx::util::placeholders::_3;

    hpx::shared_future<void> f = hpx::make_ready_future();

    std::string desc("test_bulk_then");
    {
        auto exec = hpx::experimental::prefer(
            hpx::execution::experimental::with_annotation, executor, desc);

        hpx::parallel::execution::bulk_then_execute(
            exec, hpx::util::bind(&bulk_test_f, _1, _2, _3), 107, f, 42)
            .get();

        HPX_TEST_EQ(annotation, desc);
        HPX_TEST_EQ(annotation,
            std::string(hpx::execution::experimental::get_annotation(exec)));

        annotation.clear();
        hpx::parallel::execution::bulk_then_execute(
            exec, &bulk_test_f, 107, f, 42)
            .get();

        HPX_TEST_EQ(annotation, desc);
        HPX_TEST_EQ(annotation,
            std::string(hpx::execution::experimental::get_annotation(exec)));
    }

    {
        auto exec =
            hpx::execution::experimental::with_annotation(executor, desc);

        annotation.clear();
        hpx::parallel::execution::bulk_then_execute(
            exec, hpx::util::bind(&bulk_test_f, _1, _2, _3), 107, f, 42)
            .get();

        HPX_TEST_EQ(annotation, desc);
        HPX_TEST_EQ(annotation,
            std::string(hpx::execution::experimental::get_annotation(exec)));

        annotation.clear();
        hpx::parallel::execution::bulk_then_execute(
            exec, &bulk_test_f, 107, f, 42)
            .get();

        HPX_TEST_EQ(annotation, desc);
        HPX_TEST_EQ(annotation,
            std::string(hpx::execution::experimental::get_annotation(exec)));
    }
}

///////////////////////////////////////////////////////////////////////////////
template <typename ExPolicy>
void test_post_policy(ExPolicy&& policy)
{
    std::string desc("test_post_policy");
    auto p = hpx::execution::experimental::with_annotation(policy, desc);

    hpx::lcos::local::latch l(2);
    hpx::parallel::execution::post(p.executor(), &test_post_f, 42, std::ref(l));
    l.arrive_and_wait();

    HPX_TEST_EQ(annotation, desc);
    HPX_TEST_EQ(annotation,
        std::string(hpx::execution::experimental::get_annotation(p)));
}

///////////////////////////////////////////////////////////////////////////////
template <typename ExPolicy>
void test_with_annotation(ExPolicy&& policy)
{
    test_post(policy.executor());

    test_sync(policy.executor());
    test_async(policy.executor());
    test_then(policy.executor());

    test_bulk_sync(policy.executor());
    test_bulk_async(policy.executor());
    test_bulk_then(policy.executor());

    test_post_policy(policy);
}

///////////////////////////////////////////////////////////////////////////////
void test_seq_policy()
{
    // make sure execution::seq is not used directly
    {
        auto policy = hpx::execution::experimental::with_annotation(
            hpx::execution::seq, "seq");

        static_assert(
            !std::is_same<std::decay_t<decltype(policy.executor())>,
                std::decay_t<decltype(hpx::execution::seq.executor())>>::value,
            "sequenced_executor should be wrapped in annotating_executor");
    }

    {
        auto original_policy = hpx::execution::seq;
        auto policy = hpx::execution::experimental::with_annotation(
            std::move(original_policy), "seq");

        static_assert(
            !std::is_same<std::decay_t<decltype(policy.executor())>,
                std::decay_t<decltype(hpx::execution::seq.executor())>>::value,
            "sequenced_executor should be wrapped in annotating_executor");
    }
}

void test_par_policy()
{
    // make sure execution::par is used directly
    {
        auto policy = hpx::execution::experimental::with_annotation(
            hpx::execution::par, "par");

        static_assert(
            std::is_same<std::decay_t<decltype(policy.executor())>,
                std::decay_t<decltype(hpx::execution::par.executor())>>::value,
            "parallel_executor should not be wrapped in annotating_executor");
    }

    {
        auto original_policy = hpx::execution::par;
        auto policy = hpx::execution::experimental::with_annotation(
            std::move(original_policy), "par");

        static_assert(
            std::is_same<std::decay_t<decltype(policy.executor())>,
                std::decay_t<decltype(hpx::execution::par.executor())>>::value,
            "parallel_executor should not be wrapped in annotating_executor");
    }
}

///////////////////////////////////////////////////////////////////////////////
struct test_async_executor
{
    using execution_category = hpx::execution::parallel_execution_tag;

    template <typename F, typename... Ts>
    static auto async_execute(F&& f, Ts&&... ts)
    {
        return hpx::async(std::forward<F>(f), std::forward<Ts>(ts)...);
    }
};

namespace hpx { namespace parallel { namespace execution {
    template <>
    struct is_two_way_executor<test_async_executor> : std::true_type
    {
    };
}}}    // namespace hpx::parallel::execution

int hpx_main()
{
    // supports annotations
    test_with_annotation(hpx::execution::par);

    // don't support them
    test_with_annotation(hpx::execution::seq);
    test_with_annotation(hpx::execution::par.on(test_async_executor()));

    test_seq_policy();
    test_par_policy();

    return hpx::local::finalize();
}

int main(int argc, char* argv[])
{
    // By default this test should run on all available cores
    std::vector<std::string> const cfg = {"hpx.os_threads=all"};

    // Initialize and run HPX
    hpx::local::init_params init_args;
    init_args.cfg = cfg;

    HPX_TEST_EQ_MSG(hpx::local::init(hpx_main, argc, argv, init_args), 0,
        "HPX main exited with non-zero status");

    return hpx::util::report_errors();
}
#else
int main()
{
    return 0;
}
#endif
