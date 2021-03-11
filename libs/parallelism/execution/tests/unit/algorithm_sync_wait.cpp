//  Copyright (c) 2021 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_main.hpp>
#include <hpx/modules/execution.hpp>
#include <hpx/modules/testing.hpp>

#include <atomic>
#include <exception>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

namespace ex = hpx::execution::experimental;

struct sender
{
    std::atomic<bool>& start_called;
    std::atomic<bool>& connect_called;
    std::atomic<bool>& tag_invoke_sync_wait_overload_called;

    template <template <class...> class Tuple,
        template <class...> class Variant>
    using value_types = Variant<Tuple<>>;

    template <template <class...> class Variant>
    using error_types = Variant<std::exception_ptr>;

    static constexpr bool sends_done = false;

    template <typename R>
    struct operation_state
    {
        std::atomic<bool>& start_called;
        std::decay_t<R> r;
        void start() noexcept
        {
            start_called = true;
            ex::set_value(std::move(r));
        };
    };

    template <typename R>
    auto connect(R&& r) &&
    {
        connect_called = true;
        return operation_state<R>{start_called, std::forward<R>(r)};
    }
};

struct sender2 : sender
{
    explicit sender2(sender s)
      : sender(std::move(s))
    {
    }
};

// NOTE: This is not a conforming sync_wait implementation. It only exists to
// check that the tag_invoke overload is called.
void tag_invoke(ex::sync_wait_t, sender2 s)
{
    s.tag_invoke_sync_wait_overload_called = true;
}

struct error_sender
{
    template <template <class...> class Tuple,
        template <class...> class Variant>
    using value_types = Variant<Tuple<>>;

    template <template <class...> class Variant>
    using error_types = Variant<std::exception_ptr>;

    static constexpr bool sends_done = false;

    template <typename R>
    struct operation_state
    {
        std::decay_t<R> r;
        void start() noexcept
        {
            try
            {
                throw std::runtime_error("error");
            }
            catch (...)
            {
                ex::set_error(std::move(r), std::current_exception());
            }
        };
    };

    template <typename R>
    auto connect(R&& r) &&
    {
        return operation_state<R>{std::forward<R>(r)};
    }
};

int main()
{
    // Success path
    {
        std::atomic<bool> start_called{false};
        std::atomic<bool> connect_called{false};
        std::atomic<bool> tag_invoke_sync_wait_overload_called{false};
        ex::sync_wait(sender{start_called, connect_called,
            tag_invoke_sync_wait_overload_called});
        HPX_TEST(start_called);
        HPX_TEST(connect_called);
        HPX_TEST(!tag_invoke_sync_wait_overload_called);
    }

    {
        HPX_TEST_EQ(ex::sync_wait(ex::just(3)), 3);
    }

    {
        std::atomic<bool> start_called{false};
        std::atomic<bool> connect_called{false};
        std::atomic<bool> tag_invoke_sync_wait_overload_called{false};
        ex::sync_wait(sender2{sender{start_called, connect_called,
            tag_invoke_sync_wait_overload_called}});
        HPX_TEST(!start_called);
        HPX_TEST(!connect_called);
        HPX_TEST(tag_invoke_sync_wait_overload_called);
    }

    // Failure path
    {
        bool exception_thrown = false;
        try
        {
            ex::sync_wait(error_sender{});
            HPX_TEST(false);
        }
        catch (std::runtime_error const& e)
        {
            HPX_TEST_EQ(std::string(e.what()), std::string("error"));
            exception_thrown = true;
        }
        HPX_TEST(exception_thrown);
    }

    return hpx::util::report_errors();
}
