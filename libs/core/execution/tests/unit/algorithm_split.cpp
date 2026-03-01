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

struct split_value_completion_env
{
    ex::run_loop_scheduler scheduler;

    friend ex::run_loop_scheduler tag_invoke(
        ex::get_completion_scheduler_t<ex::set_value_t>,
        split_value_completion_env const& env) noexcept
    {
        return env.scheduler;
    }
};

struct split_fast_path_scheduler_receiver
{
    ex::run_loop& loop;
    ex::run_loop_scheduler scheduler;
    std::atomic<bool>& set_value_called;

#if defined(HPX_HAVE_STDEXEC)
    using is_receiver = void;
#else
    struct is_receiver
    {
    };
#endif

    friend split_value_completion_env tag_invoke(
        ex::get_env_t, split_fast_path_scheduler_receiver const& r) noexcept
    {
        return {r.scheduler};
    }

    template <typename Error>
    friend void tag_invoke(ex::set_error_t,
        split_fast_path_scheduler_receiver&&, Error&&) noexcept
    {
        HPX_TEST(false);
    }

    friend void tag_invoke(
        ex::set_stopped_t, split_fast_path_scheduler_receiver&&) noexcept
    {
        HPX_TEST(false);
    }

    friend void tag_invoke(ex::set_value_t,
        split_fast_path_scheduler_receiver&& r) noexcept
    {
        r.set_value_called = true;
        r.loop.finish();
    }
};

// This overload is only used to check dispatching. It is not a useful
// implementation.
template <typename Allocator = hpx::util::internal_allocator<>>
auto tag_invoke(
    ex::split_t, custom_sender_tag_invoke s, Allocator const& = Allocator{})
{
    s.tag_invoke_overload_called = true;
    return void_sender{};
}

int main()
{
        // If split's predecessor completes inline during start(), add_continuation
        // takes the fast-path. This completion must still be delivered through the
        // receiver's completion scheduler and not inline.
        {
                std::atomic<bool> set_value_called{false};

                ex::run_loop loop;
                auto scheduler = loop.get_scheduler();

                auto s = ex::split(ex::just());
                auto os = ex::connect(std::move(s),
                        split_fast_path_scheduler_receiver{
                                loop, scheduler, set_value_called});
                ex::start(os);

                // If this is true here, completion was delivered inline instead of
                // being scheduled through the receiver-provided completion scheduler.
                HPX_TEST(!set_value_called.load());

                loop.run();
                HPX_TEST(set_value_called.load());
        }

    // Success path
    {
        std::atomic<bool> set_value_called{false};
        auto s1 = void_sender{};
        auto s2 = ex::split(std::move(s1));
        static_assert(ex::is_sender_v<decltype(s2)>);
#if defined(HPX_HAVE_STDEXEC)
        static_assert(ex::is_sender_in_v<decltype(s2), ex::empty_env>);
#else
        static_assert(ex::is_sender_v<decltype(s2), ex::empty_env>);
#endif

        check_value_types<hpx::variant<hpx::tuple<>>>(s2);
#if defined(HPX_HAVE_STDEXEC)
        // In p2300R8 split's error channel returns a const& of an exception_ptr
        check_error_types<hpx::variant<std::exception_ptr const&>>(s2);
#else
        check_error_types<hpx::variant<std::exception_ptr>>(s2);
#endif
        check_sends_stopped<true>(s2);

        auto f = [] {};
        auto r = callback_receiver<decltype(f)>{f, set_value_called};
        auto os = ex::connect(std::move(s2), std::move(r));
        ex::start(os);
        HPX_TEST(set_value_called);
    }

    {
        std::atomic<bool> set_value_called{false};
        auto s1 = ex::just(0);
        static_assert(ex::is_sender_v<decltype(s1)>);
#if defined(HPX_HAVE_STDEXEC)
        static_assert(ex::is_sender_in_v<decltype(s1), ex::empty_env>);
#else
        static_assert(ex::is_sender_v<decltype(s1), ex::empty_env>);
#endif

        check_value_types<hpx::variant<hpx::tuple<int>>>(s1);
#if defined(HPX_HAVE_STDEXEC)
        // In p2300R8 just does not have an error channel
        check_error_types<hpx::variant<>>(s1);
#else
        check_error_types<hpx::variant<std::exception_ptr>>(s1);
#endif
        check_sends_stopped<false>(s1);

        auto s2 = ex::split(std::move(s1));
        static_assert(ex::is_sender_v<decltype(s2)>);

        check_value_types<hpx::variant<hpx::tuple<int const&>>>(s2);
#if defined(HPX_HAVE_STDEXEC)
        check_error_types<hpx::variant<std::exception_ptr const&>>(s2);
#else
        check_error_types<hpx::variant<std::exception_ptr>>(s2);
#endif
        check_sends_stopped<true>(s2);

        auto f = [](int x) { HPX_TEST_EQ(x, 0); };
        auto r = callback_receiver<decltype(f)>{f, set_value_called};
        auto os = ex::connect(std::move(s2), std::move(r));
        ex::start(os);
        HPX_TEST(set_value_called);
    }

    {
        std::atomic<bool> set_value_called{false};
        auto s1 = ex::just(custom_type_non_default_constructible{42});
        static_assert(ex::is_sender_v<decltype(s1)>);
#if defined(HPX_HAVE_STDEXEC)
        static_assert(ex::is_sender_in_v<decltype(s1), ex::empty_env>);
#else
        static_assert(ex::is_sender_v<decltype(s1), ex::empty_env>);
#endif

        check_value_types<
            hpx::variant<hpx::tuple<custom_type_non_default_constructible>>>(
            s1);
#if defined(HPX_HAVE_STDEXEC)
        check_error_types<hpx::variant<>>(s1);
#else
        check_error_types<hpx::variant<std::exception_ptr>>(s1);
#endif
        check_sends_stopped<false>(s1);

        auto s2 = ex::split(std::move(s1));
        static_assert(ex::is_sender_v<decltype(s2)>);
#if defined(HPX_HAVE_STDEXEC)
        static_assert(ex::is_sender_in_v<decltype(s2), ex::empty_env>);
#else
        static_assert(ex::is_sender_v<decltype(s2), ex::empty_env>);
#endif

        check_value_types<hpx::variant<
            hpx::tuple<custom_type_non_default_constructible const&>>>(s2);
#if defined(HPX_HAVE_STDEXEC)
        check_error_types<hpx::variant<std::exception_ptr const&>>(s2);
#else
        check_error_types<hpx::variant<std::exception_ptr>>(s2);
#endif
        check_sends_stopped<true>(s2);

        auto f = [](auto x) { HPX_TEST_EQ(x.x, 42); };
        auto r = callback_receiver<decltype(f)>{f, set_value_called};
        auto os = ex::connect(std::move(s2), std::move(r));
        ex::start(os);
        HPX_TEST(set_value_called);
    }

    {
        std::atomic<bool> set_value_called{false};
        auto s1 =
            ex::just(custom_type_non_default_constructible_non_copyable{42});
        static_assert(ex::is_sender_v<decltype(s1)>);
#if defined(HPX_HAVE_STDEXEC)
        static_assert(ex::is_sender_in_v<decltype(s1), ex::empty_env>);
#else
        static_assert(ex::is_sender_v<decltype(s1), ex::empty_env>);
#endif

        check_value_types<hpx::variant<
            hpx::tuple<custom_type_non_default_constructible_non_copyable>>>(
            s1);
#if defined(HPX_HAVE_STDEXEC)
        check_error_types<hpx::variant<>>(s1);
#else
        check_error_types<hpx::variant<std::exception_ptr>>(s1);
#endif
        check_sends_stopped<false>(s1);

        auto s2 = ex::split(std::move(s1));
        static_assert(ex::is_sender_v<decltype(s2)>);
#if defined(HPX_HAVE_STDEXEC)
        static_assert(ex::is_sender_in_v<decltype(s2), ex::empty_env>);
#else
        static_assert(ex::is_sender_v<decltype(s2), ex::empty_env>);
#endif

        check_value_types<hpx::variant<hpx::tuple<
            custom_type_non_default_constructible_non_copyable const&>>>(s2);
#if defined(HPX_HAVE_STDEXEC)
        check_error_types<hpx::variant<std::exception_ptr const&>>(s2);
#else
        check_error_types<hpx::variant<std::exception_ptr>>(s2);
#endif
        check_sends_stopped<true>(s2);

        auto f = [](auto& x) { HPX_TEST_EQ(x.x, 42); };
        auto r = callback_receiver<decltype(f)>{f, set_value_called};
        auto os = ex::connect(std::move(s2), std::move(r));
        ex::start(os);
        HPX_TEST(set_value_called);
    }

    // operator| overload
    {
        std::atomic<bool> set_value_called{false};
        auto s = void_sender{} | ex::split();
        static_assert(ex::is_sender_v<decltype(s)>);
#if defined(HPX_HAVE_STDEXEC)
        static_assert(ex::is_sender_in_v<decltype(s), ex::empty_env>);
#else
        static_assert(ex::is_sender_v<decltype(s), ex::empty_env>);
#endif

        check_value_types<hpx::variant<hpx::tuple<>>>(s);
#if defined(HPX_HAVE_STDEXEC)
        check_error_types<hpx::variant<std::exception_ptr const&>>(s);
#else
        check_error_types<hpx::variant<std::exception_ptr>>(s);
#endif
        check_sends_stopped<true>(s);

        auto f = [] {};
        auto r = callback_receiver<decltype(f)>{f, set_value_called};
        auto os = ex::connect(std::move(s), std::move(r));
        ex::start(os);
        HPX_TEST(set_value_called);
    }

    // tag_invoke overload
    {
        std::atomic<bool> receiver_set_value_called{false};
        std::atomic<bool> tag_invoke_overload_called{false};
        auto s =
            custom_sender_tag_invoke{tag_invoke_overload_called} | ex::split();
        static_assert(ex::is_sender_v<decltype(s)>);
#if defined(HPX_HAVE_STDEXEC)
        static_assert(ex::is_sender_in_v<decltype(s), ex::empty_env>);
#else
        static_assert(ex::is_sender_v<decltype(s), ex::empty_env>);
#endif

        // custom_sender_tag_invoke implements tag_invoke(split_t, ...)
        // returning an instance of void_sender
        check_value_types<hpx::variant<hpx::tuple<>>>(s);
        check_error_types<hpx::variant<>>(s);
        check_sends_stopped<false>(s);

        auto f = [] {};
        auto r = callback_receiver<decltype(f)>{f, receiver_set_value_called};
        auto os = ex::connect(std::move(s), std::move(r));
        ex::start(os);
        HPX_TEST(receiver_set_value_called);
        HPX_TEST(tag_invoke_overload_called);
    }

    // Failure path
    {
        std::atomic<bool> set_error_called{false};
        auto s = error_sender{} | ex::split();
        static_assert(ex::is_sender_v<decltype(s)>);
#if defined(HPX_HAVE_STDEXEC)
        static_assert(ex::is_sender_in_v<decltype(s), ex::empty_env>);
#else
        static_assert(ex::is_sender_v<decltype(s), ex::empty_env>);
#endif

        check_value_types<hpx::variant<hpx::tuple<>>>(s);
#if defined(HPX_HAVE_STDEXEC)
        check_error_types<hpx::variant<std::exception_ptr const&>>(s);
#else
        check_error_types<hpx::variant<std::exception_ptr>>(s);
#endif
        check_sends_stopped<true>(s);

        auto r = error_callback_receiver<check_exception_ptr>{
            check_exception_ptr{}, set_error_called};
        auto os = ex::connect(std::move(s), std::move(r));
        ex::start(os);
        HPX_TEST(set_error_called);
    }

    {
        std::atomic<bool> set_error_called{false};
        auto s = error_sender{} | ex::split() | ex::split() | ex::split();
        static_assert(ex::is_sender_v<decltype(s)>);
#if defined(HPX_HAVE_STDEXEC)
        static_assert(ex::is_sender_in_v<decltype(s), ex::empty_env>);
#else
        static_assert(ex::is_sender_v<decltype(s), ex::empty_env>);
#endif

        check_value_types<hpx::variant<hpx::tuple<>>>(s);
#if defined(HPX_HAVE_STDEXEC)
        check_error_types<hpx::variant<std::exception_ptr const&>>(s);
#else
        check_error_types<hpx::variant<std::exception_ptr>>(s);
#endif
        check_sends_stopped<true>(s);

        auto r = error_callback_receiver<check_exception_ptr>{
            check_exception_ptr{}, set_error_called};
        auto os = ex::connect(std::move(s), std::move(r));
        ex::start(os);
        HPX_TEST(set_error_called);
    }

    // Chained split calls do not create new shared states
    {
        std::atomic<bool> receiver_set_value_called{false};
        auto s1 = ex::just() | ex::split();
        static_assert(ex::is_sender_v<decltype(s1)>);
#if defined(HPX_HAVE_STDEXEC)
        static_assert(ex::is_sender_in_v<decltype(s1), ex::empty_env>);
#else
        static_assert(ex::is_sender_v<decltype(s1), ex::empty_env>);
#endif

        check_value_types<hpx::variant<hpx::tuple<>>>(s1);
#if defined(HPX_HAVE_STDEXEC)
        check_error_types<hpx::variant<std::exception_ptr const&>>(s1);
#else
        check_error_types<hpx::variant<std::exception_ptr>>(s1);
#endif
        check_sends_stopped<true>(s1);

        auto s2 = ex::split(s1);
        static_assert(ex::is_sender_v<decltype(s2)>);
#if defined(HPX_HAVE_STDEXEC)
        static_assert(ex::is_sender_in_v<decltype(s2), ex::empty_env>);
#else
        static_assert(ex::is_sender_v<decltype(s2), ex::empty_env>);
#endif

        check_value_types<hpx::variant<hpx::tuple<>>>(s2);
#if defined(HPX_HAVE_STDEXEC)
        check_error_types<hpx::variant<std::exception_ptr const&>>(s2);
#else
        check_error_types<hpx::variant<std::exception_ptr>>(s2);
#endif
        check_sends_stopped<true>(s2);

#if !defined(HPX_HAVE_STDEXEC)
        HPX_TEST_EQ(s1.state, s2.state);
#endif
        auto s3 = ex::split(std::move(s2));
        static_assert(ex::is_sender_v<decltype(s3)>);
#if defined(HPX_HAVE_STDEXEC)
        static_assert(ex::is_sender_in_v<decltype(s3), ex::empty_env>);
#else
        static_assert(ex::is_sender_v<decltype(s3), ex::empty_env>);
#endif

        check_value_types<hpx::variant<hpx::tuple<>>>(s3);
#if defined(HPX_HAVE_STDEXEC)
        check_error_types<hpx::variant<std::exception_ptr const&>>(s3);
#else
        check_error_types<hpx::variant<std::exception_ptr>>(s3);
#endif
        check_sends_stopped<true>(s3);

#if !defined(HPX_HAVE_STDEXEC)
        HPX_TEST_EQ(s1.state, s3.state);
#endif
        auto f = [] {};
        auto r = callback_receiver<decltype(f)>{f, receiver_set_value_called};
        auto os = ex::connect(std::move(s3), std::move(r));
        ex::start(os);
        HPX_TEST(receiver_set_value_called);
    }

    {
        std::atomic<bool> receiver_set_value_called{false};
        auto s1 = ex::just(42) | ex::split();
        static_assert(ex::is_sender_v<decltype(s1)>);
#if defined(HPX_HAVE_STDEXEC)
        static_assert(ex::is_sender_in_v<decltype(s1), ex::empty_env>);
#else
        static_assert(ex::is_sender_v<decltype(s1), ex::empty_env>);
#endif

        check_value_types<hpx::variant<hpx::tuple<int const&>>>(s1);
#if defined(HPX_HAVE_STDEXEC)
        check_error_types<hpx::variant<std::exception_ptr const&>>(s1);
#else
        check_error_types<hpx::variant<std::exception_ptr>>(s1);
#endif
        check_sends_stopped<true>(s1);

        auto s2 = ex::split(s1);
        static_assert(ex::is_sender_v<decltype(s2)>);
#if defined(HPX_HAVE_STDEXEC)
        static_assert(ex::is_sender_in_v<decltype(s2), ex::empty_env>);
#else
        static_assert(ex::is_sender_v<decltype(s2), ex::empty_env>);
#endif

        check_value_types<hpx::variant<hpx::tuple<int const&>>>(s2);
#if defined(HPX_HAVE_STDEXEC)
        check_error_types<hpx::variant<std::exception_ptr const&>>(s2);
#else
        check_error_types<hpx::variant<std::exception_ptr>>(s2);
#endif
        check_sends_stopped<true>(s2);

#if !defined(HPX_HAVE_STDEXEC)
        HPX_TEST_EQ(s1.state, s2.state);
#endif
        auto s3 = ex::split(std::move(s2));
        static_assert(ex::is_sender_v<decltype(s3)>);
#if defined(HPX_HAVE_STDEXEC)
        static_assert(ex::is_sender_in_v<decltype(s3), ex::empty_env>);
#else
        static_assert(ex::is_sender_v<decltype(s3), ex::empty_env>);
#endif

        check_value_types<hpx::variant<hpx::tuple<int const&>>>(s3);
#if defined(HPX_HAVE_STDEXEC)
        check_error_types<hpx::variant<std::exception_ptr const&>>(s3);
#else
        check_error_types<hpx::variant<std::exception_ptr>>(s3);
#endif
        check_sends_stopped<true>(s3);

#if !defined(HPX_HAVE_STDEXEC)
        HPX_TEST_EQ(s1.state, s3.state);
#endif
        auto f = [](int x) { HPX_TEST_EQ(x, 42); };
        auto r = callback_receiver<decltype(f)>{f, receiver_set_value_called};
        auto os = ex::connect(std::move(s3), std::move(r));
        ex::start(os);
        HPX_TEST(receiver_set_value_called);
    }

    return hpx::util::report_errors();
}
