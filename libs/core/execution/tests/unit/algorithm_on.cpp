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

template <typename S>
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
            // set_value(self.recv_);
            ex::set_value(HPX_FORWARD(R, self.recv_));
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
            return {HPX_FORWARD(R, r)};
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

template <typename Env = ex::empty_env>
class base_expect_receiver
{
    std::atomic<bool> called_{false};
    Env env_{};

    friend Env tag_invoke(
        ex::get_env_t, const base_expect_receiver& self) noexcept
    {
        return self.env_;
    }

public:
    base_expect_receiver() = default;

    ~base_expect_receiver()
    {
        HPX_ASSERT(called_.load());
        // CHECK(called_.load());
    }

    explicit base_expect_receiver(Env env)
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

template <typename Env = ex::empty_env, typename... Ts>
struct expect_value_receiver : base_expect_receiver<Env>
{
    template <typename T = hpx::util::pack<Ts...>,
        typename = std::enable_if_t<!std::is_same_v<typename T::type, void>>>
    expect_value_receiver(Ts... vals)
      : values_(std::move(vals)...)
    {
    }

    template <HPX_CONCEPT_REQUIRES_((sizeof...(Ts) != 1))>
    explicit expect_value_receiver(Ts... vals)
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
        HPX_UNUSED(hpx::util::pack<Ts...>(vals...));
        HPX_ASSERT(self.values_ == std::tie(vals...));
        self.set_called();
    }

    template <typename... Us>
    friend void tag_invoke(
        ex::set_value_t, expect_value_receiver&&, const Us&...) noexcept
    {
        HPX_ASSERT(0 && "Should never be called");
    }

    friend void tag_invoke(ex::set_stopped_t, expect_value_receiver&&) noexcept
    {
        HPX_ASSERT(0 && "Should never be called");
    }

    template <typename E>
    friend void tag_invoke(ex::set_error_t, expect_value_receiver&&, E) noexcept
    {
        HPX_ASSERT(0 && "Should never be called");
    }

private:
    std::tuple<Ts...> values_;
};

// Borrowed from stdexec
//! Scheduler that will send impulses on user's request.
//! One can obtain senders from this, connect them to receivers and start the operation states.
//! Until the scheduler is told to start the next operation, the actions in the operation states are
//! not executed. This is similar to a task scheduler, but it's single threaded. It has basic
//! thread-safety to allow it to be run with `sync_wait` (which makes us not control when the
//! operation_state object is created and started).
struct impulse_scheduler
{
private:
    //! Command type that can store the action of firing up a sender
    using oper_command_t = std::function<void()>;
    using cmd_vec_t = std::vector<oper_command_t>;

    struct data
    {
        cmd_vec_t all_commands_;
        std::mutex mutex_;
        std::condition_variable cv_;
    };

    //! That data shared between the operation state and the actual scheduler
    //! Shared pointer to allow the scheduler to be copied (not the best semantics, but it will do)
    std::shared_ptr<data> shared_data_{};

    template <typename R>
    struct oper
    {
        data* data_;
        R receiver_;

        oper(data* shared_data, R&& recv)
          : data_(shared_data)
          , receiver_((R &&) recv)
        {
        }

        oper(oper&&) = delete;

        friend void tag_invoke(ex::start_t, oper& self) noexcept
        {
            // Enqueue another command to the list of all commands
            // The scheduler will start this, whenever start_next() is called
            std::unique_lock lock{self.data_->mutex_};
            self.data_->all_commands_.emplace_back([&self]() {
                if (ex::get_stop_token(ex::get_env(self.receiver_))
                        .stop_requested())
                {
                    ex::set_stopped((R &&) self.receiver_);
                }
                else
                {
                    ex::set_value((R &&) self.receiver_);
                }
            });
            self.data_->cv_.notify_all();
        }
    };

    struct my_sender
    {
        using id = my_sender;
        using type = my_sender;

        using is_sender = void;
        using completion_signatures = ex::completion_signatures<    //
            ex::set_value_t(),                                      //
            ex::set_stopped_t()>;
        data* shared_data_;

        template <class R>
        friend oper<std::decay_t<R>> tag_invoke(
            ex::connect_t, my_sender self, R&& r)
        {
            return {self.shared_data_, (R &&) r};
        }

        friend scheduler_env<impulse_scheduler> tag_invoke(
            ex::get_env_t, const my_sender&) noexcept
        {
            return {};
        }
    };

public:
    using id = impulse_scheduler;
    using type = impulse_scheduler;

    impulse_scheduler()
      : shared_data_(std::make_shared<data>())
    {
    }

    ~impulse_scheduler() = default;

    //! Actually start the command from the last started operation_state
    //! Blocks if no command registered (i.e., no operation state started)
    void start_next()
    {
        // Wait for a command that we can execute
        std::unique_lock lock{shared_data_->mutex_};
        while (shared_data_->all_commands_.empty())
        {
            shared_data_->cv_.wait(lock);
        }

        // Pop one command from the queue
        auto cmd = std::move(shared_data_->all_commands_.front());
        shared_data_->all_commands_.erase(shared_data_->all_commands_.begin());
        // Exit the lock before executing the command
        lock.unlock();
        // Execute the command, i.e., send an impulse to the connected sender
        cmd();
    }

    friend my_sender tag_invoke(ex::schedule_t, const impulse_scheduler& self)
    {
        return my_sender{self.shared_data_.get()};
    }

    friend bool operator==(impulse_scheduler, impulse_scheduler) noexcept
    {
        return true;
    }

    friend bool operator!=(impulse_scheduler, impulse_scheduler) noexcept
    {
        return false;
    }
};

template <typename T, typename Env = ex::empty_env>
class expect_value_receiver_ex
{
    T* dest_;
    Env env_{};

public:
    explicit expect_value_receiver_ex(T& dest)
      : dest_(&dest)
    {
    }

    expect_value_receiver_ex(Env env, T& dest)
      : dest_(&dest)
      , env_(std::move(env))
    {
    }

    friend void tag_invoke(
        ex::set_value_t, expect_value_receiver_ex self, T val) noexcept
    {
        *self.dest_ = val;
    }

    template <typename... Ts>
    friend void tag_invoke(
        ex::set_value_t, expect_value_receiver_ex, Ts...) noexcept
    {
        HPX_ASSERT(0 &&
            "set_value called with wrong value types on "
            "expect_value_receiver_ex");
    }

    friend void tag_invoke(ex::set_stopped_t, expect_value_receiver_ex) noexcept
    {
        HPX_ASSERT(0 && "set_stopped called on expect_value_receiver_ex");
    }

    friend void tag_invoke(
        ex::set_error_t, expect_value_receiver_ex, std::exception_ptr) noexcept
    {
        HPX_ASSERT(0 && "set_error called on expect_value_receiver_ex");
    }

    friend Env tag_invoke(
        ex::get_env_t, const expect_value_receiver_ex& self) noexcept
    {
        return self.env_;
    }
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

    {
        int recv_value{0};
        impulse_scheduler sched;
        auto snd = ex::on(sched, ex::just(13));
        auto op =
            ex::connect(std::move(snd), expect_value_receiver_ex{recv_value});
        ex::start(op);
        // Up until this point, the scheduler didn't start any task; no effect expected
        HPX_ASSERT(recv_value == 0);

        // Tell the scheduler to start executing one task
        sched.start_next();
        HPX_ASSERT(recv_value == 13);
    }

    return hpx::util::report_errors();
}
