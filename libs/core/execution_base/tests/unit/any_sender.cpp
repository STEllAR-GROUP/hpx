//  Copyright (c) 2021 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/execution_base/any_sender.hpp>
#include <hpx/execution_base/sender.hpp>
#include <hpx/functional/bind_front.hpp>
#include <hpx/functional/invoke_fused.hpp>
#include <hpx/modules/errors.hpp>
#include <hpx/modules/testing.hpp>

#include <atomic>
#include <exception>
#include <string>
#include <utility>

struct custom_type_non_copyable
{
    int x;

    custom_type_non_copyable(int x)
      : x(x)
    {
    }
    custom_type_non_copyable() = default;
    custom_type_non_copyable(custom_type_non_copyable&&) = default;
    custom_type_non_copyable& operator=(custom_type_non_copyable&&) = default;
    custom_type_non_copyable(custom_type_non_copyable const&) = delete;
    custom_type_non_copyable& operator=(
        custom_type_non_copyable const&) = delete;
};

template <typename... Ts>
struct non_copyable_sender
{
    hpx::tuple<std::decay_t<Ts>...> ts;

    template <template <class...> class Tuple,
        template <class...> class Variant>
    using value_types = Variant<Tuple<>>;

    template <template <class...> class Variant>
    using error_types = Variant<std::exception_ptr>;

    static constexpr bool sends_done = false;

    non_copyable_sender() = default;
    template <typename T,
        typename = std::enable_if_t<
            !std::is_same_v<std::decay_t<T>, non_copyable_sender>>>
    non_copyable_sender(T&& t)
      : ts(std::forward<T>(t))
    {
    }
    template <typename T1, typename T2, typename... Ts_>
    non_copyable_sender(T1&& t1, T2&& t2, Ts_&&... ts)
      : ts(std::forward<T1>(t1), std::forward<T2>(t2), std::forward<Ts_>(ts)...)
    {
    }
    non_copyable_sender(non_copyable_sender&&) = default;
    non_copyable_sender(non_copyable_sender const&) = delete;
    non_copyable_sender& operator=(non_copyable_sender&&) = default;
    non_copyable_sender& operator=(non_copyable_sender const&) = delete;

    template <typename R>
    struct operation_state
    {
        std::decay_t<R> r;
        hpx::tuple<std::decay_t<Ts>...> ts;

        friend void tag_dispatch(
            hpx::execution::experimental::start_t, operation_state& os) noexcept
        {
            hpx::util::invoke_fused(
                hpx::util::bind_front(
                    hpx::execution::experimental::set_value, std::move(os.r)),
                std::move(os.ts));
        };
    };

    template <typename R>
    friend operation_state<R> tag_dispatch(
        hpx::execution::experimental::connect_t, non_copyable_sender&& s,
        R&& r) noexcept
    {
        return {std::forward<R>(r), std::move(s.ts)};
    }
};

template <typename... Ts>
struct sender
{
    hpx::tuple<std::decay_t<Ts>...> ts;

    template <template <class...> class Tuple,
        template <class...> class Variant>
    using value_types = Variant<Tuple<>>;

    template <template <class...> class Variant>
    using error_types = Variant<std::exception_ptr>;

    static constexpr bool sends_done = false;

    sender() = default;
    template <typename T,
        typename = std::enable_if_t<!std::is_same_v<std::decay_t<T>, sender>>>
    sender(T&& t)
      : ts(std::forward<T>(t))
    {
    }
    template <typename T1, typename T2, typename... Ts_>
    sender(T1&& t1, T2&& t2, Ts_&&... ts)
      : ts(std::forward<T1>(t1), std::forward<T2>(t2), std::forward<Ts_>(ts)...)
    {
    }
    sender(sender&&) = default;
    sender(sender const&) = default;
    sender& operator=(sender&&) = default;
    sender& operator=(sender const&) = default;

    template <typename R>
    struct operation_state
    {
        std::decay_t<R> r;
        hpx::tuple<std::decay_t<Ts>...> ts;

        friend void tag_dispatch(
            hpx::execution::experimental::start_t, operation_state& os) noexcept
        {
            hpx::util::invoke_fused(
                hpx::util::bind_front(
                    hpx::execution::experimental::set_value, std::move(os.r)),
                std::move(os.ts));
        };
    };

    template <typename R>
    friend operation_state<R> tag_dispatch(
        hpx::execution::experimental::connect_t, sender&& s, R&& r)
    {
        return {std::forward<R>(r), std::move(s.ts)};
    }

    template <typename R>
    friend operation_state<R> tag_dispatch(
        hpx::execution::experimental::connect_t, sender& s, R&& r)
    {
        return {std::forward<R>(r), s.ts};
    }
};

template <typename... Ts>
struct large_non_copyable_sender : non_copyable_sender<Ts...>
{
    // This padding only needs to be larger than the embedded storage in
    // any_sender. Adjust if needed.
    unsigned char padding[128] = {0};

    large_non_copyable_sender() = default;
    template <typename T,
        typename = std::enable_if_t<
            !std::is_same_v<std::decay_t<T>, large_non_copyable_sender>>>
    large_non_copyable_sender(T&& t)
      : non_copyable_sender<Ts...>(std::forward<T>(t))
    {
    }
    template <typename T1, typename T2, typename... Ts_>
    large_non_copyable_sender(T1&& t1, T2&& t2, Ts_&&... ts)
      : non_copyable_sender<Ts...>(std::forward<T1>(t1), std::forward<T2>(t2),
            std::forward<Ts_>(ts)...)
    {
    }
    large_non_copyable_sender(large_non_copyable_sender&&) = default;
    large_non_copyable_sender(large_non_copyable_sender const&) = delete;
    large_non_copyable_sender& operator=(large_non_copyable_sender&&) = default;
    large_non_copyable_sender& operator=(
        large_non_copyable_sender const&) = delete;
};

template <typename... Ts>
struct large_sender : sender<Ts...>
{
    // This padding only needs to be larger than the embedded storage in
    // any_sender. Adjust if needed.
    unsigned char padding[128] = {0};

    large_sender() = default;
    template <typename T,
        typename =
            std::enable_if_t<!std::is_same_v<std::decay_t<T>, large_sender>>>
    large_sender(T&& t)
      : sender<Ts...>(std::forward<T>(t))
    {
    }
    template <typename T1, typename T2, typename... Ts_>
    large_sender(T1&& t1, T2&& t2, Ts_&&... ts)
      : sender<Ts...>(std::forward<T1>(t1), std::forward<T2>(t2),
            std::forward<Ts_>(ts)...)
    {
    }
    large_sender(large_sender&&) = default;
    large_sender(large_sender const&) = default;
    large_sender& operator=(large_sender&&) = default;
    large_sender& operator=(large_sender const&) = default;
};

struct error_sender
{
    template <template <typename...> class Tuple,
        template <typename...> class Variant>
    using value_types = Variant<Tuple<>>;

    template <template <typename...> class Variant>
    using error_types = Variant<std::exception_ptr>;

    static constexpr bool sends_done = false;

    template <typename R>
    struct operation_state
    {
        std::decay_t<R> r;
        friend void tag_dispatch(
            hpx::execution::experimental::start_t, operation_state& os) noexcept
        {
            try
            {
                throw std::runtime_error("error");
            }
            catch (...)
            {
                hpx::execution::experimental::set_error(
                    std::move(os.r), std::current_exception());
            }
        }
    };

    template <typename R>
    friend operation_state<R> tag_dispatch(
        hpx::execution::experimental::connect_t, error_sender, R&& r)
    {
        return {std::forward<R>(r)};
    }
};

template <typename F>
struct callback_receiver
{
    std::decay_t<F> f;
    std::atomic<bool>& set_value_called;

    template <typename E>
    friend void tag_dispatch(hpx::execution::experimental::set_error_t,
        callback_receiver&&, E&&) noexcept
    {
        HPX_TEST(false);
    }

    friend void tag_dispatch(
        hpx::execution::experimental::set_done_t, callback_receiver&&) noexcept
    {
        HPX_TEST(false);
    };

    template <typename... Ts>
    friend auto tag_dispatch(hpx::execution::experimental::set_value_t,
        callback_receiver&& r, Ts&&... ts) noexcept
    {
        HPX_INVOKE(std::move(r.f), std::forward<Ts>(ts)...);
        r.set_value_called = true;
    }
};

struct error_receiver
{
    std::atomic<bool>& set_error_called;

    friend void tag_dispatch(hpx::execution::experimental::set_error_t,
        error_receiver&& r, std::exception_ptr&& e) noexcept
    {
        try
        {
            std::rethrow_exception(std::move(e));
        }
        catch (std::runtime_error const& e)
        {
            HPX_TEST_EQ(std::string(e.what()), std::string("error"));
        }
        catch (...)
        {
            HPX_TEST(false);
        }
        r.set_error_called = true;
    }

    friend void tag_dispatch(
        hpx::execution::experimental::set_done_t, error_receiver&&) noexcept
    {
        HPX_TEST(false);
    };

    template <typename... Ts>
    friend void tag_dispatch(hpx::execution::experimental::set_value_t,
        error_receiver&&, Ts&&...) noexcept
    {
        HPX_TEST(false);
    }
};

template <template <typename...> typename Sender, typename... Ts, typename F>
void test_any_sender(F&& f, Ts&&... ts)
{
    namespace ex = hpx::execution::experimental;

    static_assert(std::is_copy_constructible_v<Sender<Ts...>>,
        "This test requires the sender to be copy constructible.");

    Sender<std::decay_t<Ts>...> s{std::forward<Ts>(ts)...};

    ex::any_sender<std::decay_t<Ts>...> as1{s};
    auto as2 = as1;

    // We should be able to connect both as1 and as2 multiple times; set_value
    // should always be called
    {
        std::atomic<bool> set_value_called{false};
        auto os = ex::connect(as1, callback_receiver<F>{f, set_value_called});
        ex::start(os);
        HPX_TEST(set_value_called);
    }

    {
        std::atomic<bool> set_value_called{false};
        auto os = ex::connect(
            std::move(as1), callback_receiver<F>{f, set_value_called});
        ex::start(os);
        HPX_TEST(set_value_called);
    }

    {
        std::atomic<bool> set_value_called{false};
        auto os = ex::connect(as2, callback_receiver<F>{f, set_value_called});
        ex::start(os);
        HPX_TEST(set_value_called);
    }

    {
        std::atomic<bool> set_value_called{false};
        auto os = ex::connect(
            std::move(as2), callback_receiver<F>{f, set_value_called});
        ex::start(os);
        HPX_TEST(set_value_called);
    }

    // as1 and as2 have been moved so we always expect an exception here
    {
        std::atomic<bool> set_value_called{false};
        try
        {
            auto os = ex::connect(
                // NOLINTNEXTLINE(bugprone-use-after-move)
                std::move(as1), callback_receiver<F>{f, set_value_called});
            HPX_TEST(false);
            ex::start(os);
        }
        catch (hpx::exception const& e)
        {
            HPX_TEST_EQ(e.get_error(), hpx::bad_function_call);
        }
        catch (...)
        {
            HPX_TEST(false);
        }
        HPX_TEST(!set_value_called);
    }

    {
        std::atomic<bool> set_value_called{false};
        try
        {
            auto os = ex::connect(
                // NOLINTNEXTLINE(bugprone-use-after-move)
                std::move(as2), callback_receiver<F>{f, set_value_called});
            HPX_TEST(false);
            ex::start(os);
        }
        catch (hpx::exception const& e)
        {
            HPX_TEST_EQ(e.get_error(), hpx::bad_function_call);
        }
        catch (...)
        {
            HPX_TEST(false);
        }
        HPX_TEST(!set_value_called);
    }
}

template <template <typename...> typename Sender, typename... Ts, typename F>
void test_unique_any_sender(F&& f, Ts&&... ts)
{
    namespace ex = hpx::execution::experimental;

    Sender<std::decay_t<Ts>...> s{std::forward<Ts>(ts)...};

    ex::unique_any_sender<std::decay_t<Ts>...> as1{std::move(s)};
    auto as2 = std::move(as1);

    // We expect set_value to be called here
    {
        std::atomic<bool> set_value_called = false;
        auto os = ex::connect(
            std::move(as2), callback_receiver<F>{f, set_value_called});
        ex::start(os);
        HPX_TEST(set_value_called);
    }

    // as1 has been moved so we always expect an exception here
    {
        std::atomic<bool> set_value_called{false};
        try
        {
            auto os = ex::connect(
                // NOLINTNEXTLINE(bugprone-use-after-move)
                std::move(as1), callback_receiver<F>{f, set_value_called});
            HPX_TEST(false);
            ex::start(os);
        }
        catch (hpx::exception const& e)
        {
            HPX_TEST_EQ(e.get_error(), hpx::bad_function_call);
        }
        catch (...)
        {
            HPX_TEST(false);
        }
        HPX_TEST(!set_value_called);
    }
}

void test_any_sender_set_error()
{
    namespace ex = hpx::execution::experimental;

    error_sender s;

    ex::any_sender<> as1{std::move(s)};
    auto as2 = as1;

    // We should be able to connect the sender multiple times; set_error should
    // always be called
    {
        std::atomic<bool> set_error_called{false};
        auto os = ex::connect(as1, error_receiver{set_error_called});
        ex::start(os);
        HPX_TEST(set_error_called);
    }

    {
        std::atomic<bool> set_error_called{false};
        auto os = ex::connect(std::move(as1), error_receiver{set_error_called});
        ex::start(os);
        HPX_TEST(set_error_called);
    }

    {
        std::atomic<bool> set_error_called{false};
        auto os = ex::connect(as2, error_receiver{set_error_called});
        ex::start(os);
        HPX_TEST(set_error_called);
    }

    {
        std::atomic<bool> set_error_called{false};
        auto os = ex::connect(std::move(as2), error_receiver{set_error_called});
        ex::start(os);
        HPX_TEST(set_error_called);
    }

    // as1 and as2 have been moved so we always expect an exception here
    {
        std::atomic<bool> set_error_called{false};
        try
        {
            auto os =
                // NOLINTNEXTLINE(bugprone-use-after-move)
                ex::connect(std::move(as1), error_receiver{set_error_called});
            HPX_TEST(false);
            ex::start(os);
        }
        catch (hpx::exception const& e)
        {
            HPX_TEST_EQ(e.get_error(), hpx::bad_function_call);
        }
        catch (...)
        {
            HPX_TEST(false);
        }
        HPX_TEST(!set_error_called);
    }

    {
        std::atomic<bool> set_error_called{false};
        try
        {
            auto os =
                // NOLINTNEXTLINE(bugprone-use-after-move)
                ex::connect(std::move(as2), error_receiver{set_error_called});
            HPX_TEST(false);
            ex::start(os);
        }
        catch (hpx::exception const& e)
        {
            HPX_TEST_EQ(e.get_error(), hpx::bad_function_call);
        }
        catch (...)
        {
            HPX_TEST(false);
        }
        HPX_TEST(!set_error_called);
    }
}

void test_unique_any_sender_set_error()
{
    namespace ex = hpx::execution::experimental;

    error_sender s;

    ex::unique_any_sender<> as1{std::move(s)};
    auto as2 = std::move(as1);

    // We expect set_error to be called here
    {
        std::atomic<bool> set_error_called{false};
        auto os = ex::connect(std::move(as2), error_receiver{set_error_called});
        ex::start(os);
        HPX_TEST(set_error_called);
    }

    // as1 has been moved so we always expect an exception here
    {
        std::atomic<bool> set_error_called{false};
        try
        {
            auto os =
                // NOLINTNEXTLINE(bugprone-use-after-move)
                ex::connect(std::move(as1), error_receiver{set_error_called});
            HPX_TEST(false);
            ex::start(os);
        }
        catch (hpx::exception const& e)
        {
            HPX_TEST_EQ(e.get_error(), hpx::bad_function_call);
        }
        catch (...)
        {
            HPX_TEST(false);
        }
        HPX_TEST(!set_error_called);
    }
}

int main()
{
    // We can only wrap copyable senders in any_sender
    test_any_sender<sender>([] {});
    test_any_sender<sender, int>([](int x) { HPX_TEST_EQ(x, 42); }, 42);
    test_any_sender<sender, int, double>(
        [](int x, double y) {
            HPX_TEST_EQ(x, 42);
            HPX_TEST_EQ(y, 3.14);
        },
        42, 3.14);

    test_any_sender<large_sender>([] {});
    test_any_sender<large_sender, int>([](int x) { HPX_TEST_EQ(x, 42); }, 42);
    test_any_sender<large_sender, int, double>(
        [](int x, double y) {
            HPX_TEST_EQ(x, 42);
            HPX_TEST_EQ(y, 3.14);
        },
        42, 3.14);

    // We can wrap both copyable and non-copyable senders in unique_any_sender
    test_unique_any_sender<sender>([] {});
    test_unique_any_sender<sender, int>([](int x) { HPX_TEST_EQ(x, 42); }, 42);
    test_unique_any_sender<sender, int, double>(
        [](int x, double y) {
            HPX_TEST_EQ(x, 42);
            HPX_TEST_EQ(y, 3.14);
        },
        42, 3.14);

    test_unique_any_sender<large_sender>([] {});
    test_unique_any_sender<large_sender, int>(
        [](int x) { HPX_TEST_EQ(x, 42); }, 42);
    test_unique_any_sender<large_sender, int, double>(
        [](int x, double y) {
            HPX_TEST_EQ(x, 42);
            HPX_TEST_EQ(y, 3.14);
        },
        42, 3.14);

    test_unique_any_sender<non_copyable_sender>([] {});
    test_unique_any_sender<non_copyable_sender, int>(
        [](int x) { HPX_TEST_EQ(x, 42); }, 42);
    test_unique_any_sender<non_copyable_sender, int, double>(
        [](int x, double y) {
            HPX_TEST_EQ(x, 42);
            HPX_TEST_EQ(y, 3.14);
        },
        42, 3.14);
    test_unique_any_sender<non_copyable_sender, int, double,
        custom_type_non_copyable>(
        [](int x, double y, custom_type_non_copyable z) {
            HPX_TEST_EQ(x, 42);
            HPX_TEST_EQ(y, 3.14);
            HPX_TEST_EQ(z.x, 43);
        },
        42, 3.14, custom_type_non_copyable(43));

    test_unique_any_sender<large_non_copyable_sender>([] {});
    test_unique_any_sender<large_non_copyable_sender, int>(
        [](int x) { HPX_TEST_EQ(x, 42); }, 42);
    test_unique_any_sender<large_non_copyable_sender, int, double>(
        [](int x, double y) {
            HPX_TEST_EQ(x, 42);
            HPX_TEST_EQ(y, 3.14);
        },
        42, 3.14);
    test_unique_any_sender<large_non_copyable_sender, int, double,
        custom_type_non_copyable>(
        [](int x, double y, custom_type_non_copyable z) {
            HPX_TEST_EQ(x, 42);
            HPX_TEST_EQ(y, 3.14);
            HPX_TEST_EQ(z.x, 43);
        },
        42, 3.14, custom_type_non_copyable(43));

    // Failure paths
    test_any_sender_set_error();
    test_unique_any_sender_set_error();

    return hpx::util::report_errors();
}
