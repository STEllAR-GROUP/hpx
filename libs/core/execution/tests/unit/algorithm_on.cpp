//  Copyright (c) 2023 Shreyas Atre
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/modules/assertion.hpp>
#include <hpx/modules/execution.hpp>
#include <hpx/modules/testing.hpp>
#include <hpx/type_support/meta.hpp>

#include <exception>
#include <type_traits>
#include <utility>

namespace ex = hpx::execution::experimental;

template <class S>
struct scheduler_env
{
    template <typename CPO>
    friend S tag_invoke(
        ex::get_completion_scheduler_t<CPO>, const scheduler_env&) noexcept
    {
        return {};
    }
};

//! Scheduler that executes everything inline, i.e., on the same thread
struct inline_scheduler
{
    using id = inline_scheduler;
    using type = inline_scheduler;
    template <typename R>
    struct operation
    {
        R recv_;

        friend void tag_invoke(ex::start_t, operation& self) noexcept
        {
            ex::set_value((R &&) self.recv_);
        }
    };

    struct my_sender
    {
        using is_sender = void;
        using completion_signatures =
            ex::completion_signatures<ex::set_value_t()>;

        template <typename R>
        friend operation<R> tag_invoke(ex::connect_t, my_sender, R&& r)
        {
            return {{}, (R &&) r};
        }

        friend scheduler_env<inline_scheduler> tag_invoke(
            ex::get_env_t, const my_sender&) noexcept
        {
            return {};
        }
    };

    friend my_sender tag_invoke(ex::schedule_t, inline_scheduler)
    {
        return {};
    }

    friend bool operator==(inline_scheduler, inline_scheduler) noexcept
    {
        return true;
    }

    friend bool operator!=(inline_scheduler, inline_scheduler) noexcept
    {
        return false;
    }
};

template <class _Env = ex::empty_env>
class base_expect_receiver
{
    std::atomic<bool> called_{false};
    _Env env_{};

    friend _Env tag_invoke(
        ex::get_env_t, const base_expect_receiver& self) noexcept
    {
        return self.env_;
    }

public:
    base_expect_receiver() = default;

    ~base_expect_receiver()
    {
        called_.load();
        // CHECK(called_.load());
    }

    explicit base_expect_receiver(_Env env)
      : env_(std::move(env))
    {
    }

    base_expect_receiver(base_expect_receiver&& other)
      : called_(other.called_.exchange(true))
      , env_(std::move(other.env_))
    {
    }

    base_expect_receiver& operator=(base_expect_receiver&& other)
    {
        called_.store(other.called_.exchange(true));
        env_ = std::move(other.env_);
        return *this;
    }

    void set_called()
    {
        called_.store(true);
    }
};

struct env_tag
{
};

template <class Env = ex::empty_env, typename... Ts>
struct expect_value_receiver : base_expect_receiver<Env>
{
    explicit(sizeof...(Ts) != 1) expect_value_receiver(Ts... vals)
      : values_(std::move(vals)...)
    {
    }

    expect_value_receiver(env_tag, Env env, Ts... vals)
      : base_expect_receiver<Env>(std::move(env))
      , values_(std::move(vals)...)
    {
    }

    friend void tag_invoke(ex::set_value_t, expect_value_receiver&& self,
        const Ts&... vals) noexcept
    {
        // CHECK(self.values_ == std::tie(vals...));
        HPX_ASSERT(self.values_ == std::tie(vals...));
        self.set_called();
    }

    template <typename... Us>
    friend void tag_invoke(
        ex::set_value_t, expect_value_receiver&&, const Us&...) noexcept
    {
        // FAIL_CHECK("set_value called with wrong value types on expect_value_receiver");
    }

    friend void tag_invoke(ex::set_stopped_t, expect_value_receiver&&) noexcept
    {
        // FAIL_CHECK("set_stopped called on expect_value_receiver");
    }

    template <typename E>
    friend void tag_invoke(ex::set_error_t, expect_value_receiver&&, E) noexcept
    {
        // FAIL_CHECK("set_error called on expect_value_receiver");
    }

private:
    std::tuple<Ts...> values_;
};

int main()
{
    {
        auto snd = ex::on(inline_scheduler{}, ex::just(13));
        static_assert(ex::is_sender_v<decltype(snd)>);
        (void) snd;
    }

    {
        auto snd = ex::on(inline_scheduler{}, ex::just(13));
        static_assert(ex::is_sender_v<decltype(snd), ex::empty_env>);
        (void) snd;
    }

    {
        auto snd = ex::on(inline_scheduler{}, ex::just(13));
        auto op = ex::connect(std::move(snd), expect_value_receiver{13});
        ex::start(op);
    }
    return hpx::util::report_errors();
}
