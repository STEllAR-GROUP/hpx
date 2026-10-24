//  Copyright (c) 2026 Vansh Dobhal
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// This test checks hpx::threads::with_task_size: the size is visible through
// traits::get_task_size (also through annotated_function and deferred_call),
// the wrapper forwards arguments and results, and it runs through hpx::async,
// hpx::post and an executor.

#include <hpx/execution.hpp>
#include <hpx/functional.hpp>
#include <hpx/future.hpp>
#include <hpx/init.hpp>
#include <hpx/latch.hpp>
#include <hpx/modules/testing.hpp>

#include <cstring>
#include <memory>
#include <string>
#include <utility>

using hpx::threads::task_size;
using hpx::threads::with_task_size;

namespace {

    int plain_function(int i)
    {
        return i + 1;
    }

    template <typename F>
    inline constexpr task_size size_of = hpx::traits::get_task_size<F>::value;

    template <typename F>
    constexpr task_size size_of_object(F const&)
    {
        return size_of<F>;
    }
}    // namespace

///////////////////////////////////////////////////////////////////////////////
void test_trait()
{
    auto lambda = [](int i) { return i; };
    static_assert(size_of<decltype(lambda)> == task_size::unknown);
    static_assert(size_of<decltype(&plain_function)> == task_size::unknown);

    auto large = with_task_size<task_size::large>(lambda);
    static_assert(size_of<decltype(large)> == task_size::large);
    static_assert(size_of_object(with_task_size<task_size::small_>(
                      &plain_function)) == task_size::small_);
    static_assert(size_of_object(with_task_size<task_size::medium>(lambda)) ==
        task_size::medium);

    // the size is a template argument, so the wrapper is as big as f
    static_assert(sizeof(large) == sizeof(lambda));

    // hpx::async binds its arguments with deferred_call
    static_assert(size_of<decltype(hpx::util::deferred_call(large, 1))> ==
        task_size::large);
    static_assert(size_of<decltype(hpx::util::deferred_call(lambda, 1))> ==
        task_size::unknown);
}

///////////////////////////////////////////////////////////////////////////////
void test_forwarding()
{
    auto add = with_task_size<task_size::small_>(&plain_function);
    HPX_TEST_EQ(add(41), 42);

    auto take = with_task_size<task_size::medium>(
        [](std::unique_ptr<int> p) { return p; });
    std::unique_ptr<int> p = take(std::make_unique<int>(7));
    HPX_TEST(p != nullptr);
    HPX_TEST_EQ(*p, 7);

    // a mutable function keeps its state between calls
    auto count =
        with_task_size<task_size::small_>([n = 0]() mutable { return ++n; });
    count();
    HPX_TEST_EQ(count(), 2);

    // lvalue arguments are passed through as references
    int value = 0;
    auto set = with_task_size<task_size::small_>([](int& i) { i = 5; });
    set(value);
    HPX_TEST_EQ(value, 5);
}

///////////////////////////////////////////////////////////////////////////////
void test_launch()
{
    auto f = with_task_size<task_size::large>(
        [](std::unique_ptr<int> p, int i) { return *p + i; });

    HPX_TEST_EQ(hpx::async(f, std::make_unique<int>(40), 2).get(), 42);
    HPX_TEST_EQ(
        hpx::async(hpx::launch::fork, f, std::make_unique<int>(1), 1).get(), 2);

    hpx::latch l(2);
    int result = 0;
    hpx::post(with_task_size<task_size::small_>([&](int i) {
        result = i;
        l.count_down(1);
    }),
        3);
    l.arrive_and_wait();
    HPX_TEST_EQ(result, 3);

    hpx::execution::parallel_executor exec;
    HPX_TEST_EQ(hpx::async(exec, f, std::make_unique<int>(10), 5).get(), 15);
    HPX_TEST_EQ(hpx::parallel::execution::sync_execute(
                    exec, f, std::make_unique<int>(1), 2),
        3);
}

///////////////////////////////////////////////////////////////////////////////
void test_composition()
{
    auto f = [] { return 1; };

    auto sized_annotated =
        with_task_size<task_size::large>(hpx::annotated_function(f, "outer"));
    auto annotated_sized =
        hpx::annotated_function(with_task_size<task_size::medium>(f), "inner");

    static_assert(size_of_object(sized_annotated) == task_size::large);
    static_assert(size_of_object(annotated_sized) == task_size::medium);
    static_assert(
        size_of<decltype(hpx::util::deferred_call(sized_annotated))> ==
        task_size::large);

#if defined(HPX_HAVE_THREAD_DESCRIPTION)
    using sized_annotated_type = decltype(sized_annotated);
    using annotated_sized_type = decltype(annotated_sized);

    HPX_TEST_EQ(
        std::strcmp(
            hpx::traits::get_function_annotation<sized_annotated_type>::call(
                sized_annotated),
            "outer"),
        0);
    HPX_TEST_EQ(
        std::strcmp(
            hpx::traits::get_function_annotation<annotated_sized_type>::call(
                annotated_sized),
            "inner"),
        0);

    // the name also reaches the thread the task runs on
    auto name_of_thread = [] {
        return std::string(
            hpx::threads::get_thread_description(hpx::threads::get_self_id())
                .get_description());
    };
    HPX_TEST_EQ(
        hpx::async(with_task_size<task_size::large>(
                       hpx::annotated_function(name_of_thread, "outer")))
            .get(),
        std::string("outer"));
    HPX_TEST_EQ(
        hpx::async(
            hpx::annotated_function(
                with_task_size<task_size::medium>(name_of_thread), "inner"))
            .get(),
        std::string("inner"));
#endif

    HPX_TEST_EQ(sized_annotated(), 1);
    HPX_TEST_EQ(annotated_sized(), 1);
}

///////////////////////////////////////////////////////////////////////////////
int hpx_main()
{
    test_trait();
    test_forwarding();
    test_launch();
    test_composition();

    return hpx::local::finalize();
}

int main(int argc, char* argv[])
{
    HPX_TEST_EQ(hpx::local::init(hpx_main, argc, argv), 0);
    return hpx::util::report_errors();
}
