//  Copyright (c) 2021 ETH Zurich
//  Copyright (c) 2022 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/modules/execution.hpp>
#include <hpx/modules/testing.hpp>

#include "algorithm_test_utils.hpp"

#include <atomic>
#include <exception>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

namespace ex = hpx::execution::experimental;

// schedule_from customization
struct scheduler_schedule_from : scheduler
{
    explicit scheduler_schedule_from(scheduler sched)
      : scheduler(std::move(sched))
    {
    }
};

template <typename Sender>
auto tag_invoke(ex::schedule_from_t, scheduler_schedule_from sched, Sender&&)
{
    sched.tag_invoke_overload_called = true;
    return scheduler::sender{};
}

// transfer customization
struct scheduler_transfer : scheduler
{
    explicit scheduler_transfer(scheduler sched)
      : scheduler(std::move(sched))
    {
    }
};

template <typename Sender, typename Scheduler>
decltype(auto) tag_invoke(ex::transfer_t, scheduler_transfer completion_sched,
    Sender&& sender, Scheduler&& sched)
{
    completion_sched.tag_invoke_overload_called = true;
    return ex::schedule_from(
        std::forward<Scheduler>(sched), std::forward<Sender>(sender));
}

struct sender_with_completion_scheduler : void_sender
{
    scheduler_transfer sched;

    explicit sender_with_completion_scheduler(scheduler_transfer sched)
      : sched(std::move(sched))
    {
    }

    friend scheduler_transfer tag_invoke(
        ex::get_completion_scheduler_t<ex::set_value_t>,
        sender_with_completion_scheduler s)
    {
        return s.sched;
    }

    template <typename Env>
    friend auto tag_invoke(
        hpx::execution::experimental::get_completion_signatures_t,
        sender_with_completion_scheduler const&, Env)
        -> hpx::execution::experimental::completion_signatures<
            hpx::execution::experimental::set_value_t()>;
};

int main()
{
    // Success path
    {
        std::atomic<bool> set_value_called{false};
        std::atomic<bool> scheduler_schedule_called{false};
        std::atomic<bool> scheduler_execute_called{false};
        std::atomic<bool> tag_invoke_overload_called{false};
        auto s = ex::transfer(ex::just(),
            scheduler{scheduler_schedule_called, scheduler_execute_called,
                tag_invoke_overload_called});
        static_assert(ex::is_sender_v<decltype(s)>);
        static_assert(ex::is_sender_v<decltype(s), ex::empty_env>);

        check_value_types<hpx::variant<hpx::tuple<>>>(s);
        check_error_types<hpx::variant<std::exception_ptr>>(s);
        check_sends_stopped<false>(s);

        auto f = [] {};
        auto r = callback_receiver<decltype(f)>{f, set_value_called};
        auto os = ex::connect(std::move(s), std::move(r));
        ex::start(os);
        HPX_TEST(set_value_called);
        HPX_TEST(!tag_invoke_overload_called);
        HPX_TEST(scheduler_schedule_called);
        HPX_TEST(!scheduler_execute_called);
    }

    {
        std::atomic<bool> set_value_called{false};
        std::atomic<bool> scheduler_schedule_called{false};
        std::atomic<bool> scheduler_execute_called{false};
        std::atomic<bool> tag_invoke_overload_called{false};
        auto s = ex::transfer(ex::just(3),
            scheduler{scheduler_schedule_called, scheduler_execute_called,
                tag_invoke_overload_called});
        static_assert(ex::is_sender_v<decltype(s)>);
        static_assert(ex::is_sender_v<decltype(s), ex::empty_env>);

        check_value_types<hpx::variant<hpx::tuple<int>>>(s);
        check_error_types<hpx::variant<std::exception_ptr>>(s);
        check_sends_stopped<false>(s);

        auto f = [](int x) { HPX_TEST_EQ(x, 3); };
        auto r = callback_receiver<decltype(f)>{f, set_value_called};
        auto os = ex::connect(std::move(s), std::move(r));
        ex::start(os);
        HPX_TEST(set_value_called);
        HPX_TEST(!tag_invoke_overload_called);
        HPX_TEST(scheduler_schedule_called);
        HPX_TEST(!scheduler_execute_called);
    }

    {
        std::atomic<bool> set_value_called{false};
        std::atomic<bool> scheduler_schedule_called{false};
        std::atomic<bool> scheduler_execute_called{false};
        std::atomic<bool> tag_invoke_overload_called{false};
        auto s =
            ex::transfer(ex::just(custom_type_non_default_constructible{42}),
                scheduler{scheduler_schedule_called, scheduler_execute_called,
                    tag_invoke_overload_called});
        static_assert(ex::is_sender_v<decltype(s)>);
        static_assert(ex::is_sender_v<decltype(s), ex::empty_env>);

        check_value_types<
            hpx::variant<hpx::tuple<custom_type_non_default_constructible>>>(s);
        check_error_types<hpx::variant<std::exception_ptr>>(s);
        check_sends_stopped<false>(s);

        auto f = [](auto x) { HPX_TEST_EQ(x.x, 42); };
        auto r = callback_receiver<decltype(f)>{f, set_value_called};
        auto os = ex::connect(std::move(s), std::move(r));
        ex::start(os);
        HPX_TEST(set_value_called);
        HPX_TEST(!tag_invoke_overload_called);
        HPX_TEST(scheduler_schedule_called);
        HPX_TEST(!scheduler_execute_called);
    }

    {
        std::atomic<bool> set_value_called{false};
        std::atomic<bool> scheduler_schedule_called{false};
        std::atomic<bool> scheduler_execute_called{false};
        std::atomic<bool> tag_invoke_overload_called{false};
        auto s = ex::transfer(
            ex::just(custom_type_non_default_constructible_non_copyable{42}),
            scheduler{scheduler_schedule_called, scheduler_execute_called,
                tag_invoke_overload_called});
        static_assert(ex::is_sender_v<decltype(s)>);
        static_assert(ex::is_sender_v<decltype(s), ex::empty_env>);

        check_value_types<hpx::variant<
            hpx::tuple<custom_type_non_default_constructible_non_copyable>>>(s);
        check_error_types<hpx::variant<std::exception_ptr>>(s);
        check_sends_stopped<false>(s);

        auto f = [](auto x) { HPX_TEST_EQ(x.x, 42); };
        auto r = callback_receiver<decltype(f)>{f, set_value_called};
        auto os = ex::connect(std::move(s), std::move(r));
        ex::start(os);
        HPX_TEST(set_value_called);
        HPX_TEST(!tag_invoke_overload_called);
        HPX_TEST(scheduler_schedule_called);
        HPX_TEST(!scheduler_execute_called);
    }

    {
        std::atomic<bool> set_value_called{false};
        std::atomic<bool> scheduler_execute_called{false};
        std::atomic<bool> scheduler_schedule_called{false};
        std::atomic<bool> tag_invoke_overload_called{false};
        auto s = ex::transfer(ex::just(std::string("hello"), 3),
            scheduler{scheduler_schedule_called, scheduler_execute_called,
                tag_invoke_overload_called});
        static_assert(ex::is_sender_v<decltype(s)>);
        static_assert(ex::is_sender_v<decltype(s), ex::empty_env>);

        check_value_types<hpx::variant<hpx::tuple<std::string, int>>>(s);
        check_error_types<hpx::variant<std::exception_ptr>>(s);
        check_sends_stopped<false>(s);

        auto f = [](std::string s, int x) {
            HPX_TEST_EQ(s, std::string("hello"));
            HPX_TEST_EQ(x, 3);
        };
        auto r = callback_receiver<decltype(f)>{f, set_value_called};
        auto os = ex::connect(std::move(s), std::move(r));
        ex::start(os);
        HPX_TEST(set_value_called);
        HPX_TEST(!tag_invoke_overload_called);
        HPX_TEST(!scheduler_execute_called);
        HPX_TEST(scheduler_schedule_called);
    }

    // operator| overload
    {
        std::atomic<bool> set_value_called{false};
        std::atomic<bool> scheduler_execute_called{false};
        std::atomic<bool> scheduler_schedule_called{false};
        std::atomic<bool> tag_invoke_overload_called{false};
        auto s = ex::just(std::string("hello"), 3) |
            ex::transfer(scheduler{scheduler_schedule_called,
                scheduler_execute_called, tag_invoke_overload_called});
        static_assert(ex::is_sender_v<decltype(s)>);
        static_assert(ex::is_sender_v<decltype(s), ex::empty_env>);

        check_value_types<hpx::variant<hpx::tuple<std::string, int>>>(s);
        check_error_types<hpx::variant<std::exception_ptr>>(s);
        check_sends_stopped<false>(s);

        auto f = [](std::string s, int x) {
            HPX_TEST_EQ(s, std::string("hello"));
            HPX_TEST_EQ(x, 3);
        };
        auto r = callback_receiver<decltype(f)>{f, set_value_called};
        auto os = ex::connect(std::move(s), r);
        ex::start(os);
        HPX_TEST(set_value_called);
        HPX_TEST(!tag_invoke_overload_called);
        HPX_TEST(scheduler_schedule_called);
        HPX_TEST(!scheduler_execute_called);
    }

    // tag_invoke overloads
    {
        std::atomic<bool> set_value_called{false};
        std::atomic<bool> tag_invoke_overload_called{false};
        std::atomic<bool> scheduler_schedule_called{false};
        std::atomic<bool> scheduler_execute_called{false};
        auto s = ex::transfer(ex::just(),
            scheduler_schedule_from{scheduler{scheduler_schedule_called,
                scheduler_execute_called, tag_invoke_overload_called}});
        static_assert(ex::is_sender_v<decltype(s)>);
        static_assert(ex::is_sender_v<decltype(s), ex::empty_env>);

        check_value_types<hpx::variant<hpx::tuple<>>>(s);
        check_error_types<hpx::variant<std::exception_ptr>>(s);
        check_sends_stopped<false>(s);

        auto f = [] {};
        auto r = callback_receiver<decltype(f)>{f, set_value_called};
        auto os = ex::connect(std::move(s), std::move(r));
        ex::start(os);
        HPX_TEST(set_value_called);
        HPX_TEST(tag_invoke_overload_called);
        HPX_TEST(!scheduler_schedule_called);
        HPX_TEST(!scheduler_execute_called);
    }

    {
        std::atomic<bool> set_value_called{false};
        std::atomic<bool> source_scheduler_tag_invoke_overload_called{false};
        std::atomic<bool> source_scheduler_schedule_called{false};
        std::atomic<bool> source_scheduler_execute_called{false};
        std::atomic<bool> destination_scheduler_tag_invoke_overload_called{
            false};
        std::atomic<bool> destination_scheduler_schedule_called{false};
        std::atomic<bool> destination_scheduler_execute_called{false};

        scheduler_transfer source_scheduler{scheduler{
            source_scheduler_schedule_called, source_scheduler_execute_called,
            source_scheduler_tag_invoke_overload_called}};
        scheduler destination_scheduler{
            scheduler{destination_scheduler_schedule_called,
                destination_scheduler_execute_called,
                destination_scheduler_tag_invoke_overload_called}};

        auto s = ex::transfer(
            sender_with_completion_scheduler{std::move(source_scheduler)},
            destination_scheduler);
        static_assert(ex::is_sender_v<decltype(s)>);
        static_assert(ex::is_sender_v<decltype(s), ex::empty_env>);

        check_value_types<hpx::variant<hpx::tuple<>>>(s);
        check_error_types<hpx::variant<std::exception_ptr>>(s);
        check_sends_stopped<false>(s);

        auto f = [] {};
        auto r = callback_receiver<decltype(f)>{f, set_value_called};
        auto os = ex::connect(std::move(s), std::move(r));
        ex::start(os);
        HPX_TEST(set_value_called);
        HPX_TEST(source_scheduler_tag_invoke_overload_called);
        HPX_TEST(!source_scheduler_schedule_called);
        HPX_TEST(!source_scheduler_execute_called);
        HPX_TEST(!destination_scheduler_tag_invoke_overload_called);
        HPX_TEST(destination_scheduler_schedule_called);
        HPX_TEST(!destination_scheduler_execute_called);
    }

    {
        std::atomic<bool> set_value_called{false};
        std::atomic<bool> source_scheduler_tag_invoke_overload_called{false};
        std::atomic<bool> source_scheduler_schedule_called{false};
        std::atomic<bool> source_scheduler_execute_called{false};
        std::atomic<bool> destination_scheduler_tag_invoke_overload_called{
            false};
        std::atomic<bool> destination_scheduler_schedule_called{false};
        std::atomic<bool> destination_scheduler_execute_called{false};

        scheduler_transfer source_scheduler{scheduler{
            source_scheduler_schedule_called, source_scheduler_execute_called,
            source_scheduler_tag_invoke_overload_called}};
        scheduler_schedule_from destination_scheduler{
            scheduler{destination_scheduler_schedule_called,
                destination_scheduler_execute_called,
                destination_scheduler_tag_invoke_overload_called}};

        auto s = ex::transfer(
            sender_with_completion_scheduler{std::move(source_scheduler)},
            destination_scheduler);
        static_assert(ex::is_sender_v<decltype(s)>);
        static_assert(ex::is_sender_v<decltype(s), ex::empty_env>);

        check_value_types<hpx::variant<hpx::tuple<>>>(s);
        check_error_types<hpx::variant<std::exception_ptr>>(s);
        check_sends_stopped<false>(s);

        auto f = [] {};
        auto r = callback_receiver<decltype(f)>{f, set_value_called};
        auto os = ex::connect(std::move(s), std::move(r));
        ex::start(os);
        HPX_TEST(set_value_called);
        HPX_TEST(source_scheduler_tag_invoke_overload_called);
        HPX_TEST(!source_scheduler_schedule_called);
        HPX_TEST(!source_scheduler_execute_called);
        HPX_TEST(destination_scheduler_tag_invoke_overload_called);
        HPX_TEST(!destination_scheduler_schedule_called);
        HPX_TEST(!destination_scheduler_execute_called);
    }

    // Failure path
    {
        std::atomic<bool> set_error_called{false};
        std::atomic<bool> tag_invoke_overload_called{false};
        std::atomic<bool> scheduler_schedule_called{false};
        std::atomic<bool> scheduler_execute_called{false};
        auto s = ex::transfer(error_sender{},
            scheduler{scheduler_schedule_called, scheduler_execute_called,
                tag_invoke_overload_called});
        static_assert(ex::is_sender_v<decltype(s)>);
        static_assert(ex::is_sender_v<decltype(s), ex::empty_env>);

        check_value_types<hpx::variant<hpx::tuple<>>>(s);
        check_error_types<hpx::variant<std::exception_ptr>>(s);
        check_sends_stopped<false>(s);

        auto r = error_callback_receiver<check_exception_ptr>{
            check_exception_ptr{}, set_error_called};
        auto os = ex::connect(std::move(s), std::move(r));
        ex::start(os);
        HPX_TEST(set_error_called);
        HPX_TEST(!tag_invoke_overload_called);
        HPX_TEST(!scheduler_schedule_called);
        HPX_TEST(!scheduler_execute_called);
    }

    return hpx::util::report_errors();
}
