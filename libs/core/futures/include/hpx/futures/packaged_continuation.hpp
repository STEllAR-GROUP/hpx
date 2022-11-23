//  Copyright (c) 2007-2021 Hartmut Kaiser
//  Copyright (c) 2014-2015 Agustin Berge
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/allocator_support/allocator_deleter.hpp>
#include <hpx/allocator_support/internal_allocator.hpp>
#include <hpx/async_base/launch_policy.hpp>
#include <hpx/errors/try_catch_exception_ptr.hpp>
#include <hpx/futures/detail/future_data.hpp>
#include <hpx/futures/traits/acquire_shared_state.hpp>
#include <hpx/futures/traits/future_access.hpp>
#include <hpx/futures/traits/future_traits.hpp>
#include <hpx/modules/errors.hpp>
#include <hpx/modules/memory.hpp>
#include <hpx/threading_base/annotated_function.hpp>
#include <hpx/threading_base/thread_description.hpp>

#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <type_traits>
#include <utility>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace lcos { namespace detail {

    template <typename Future, typename SourceState, typename DestinationState>
    HPX_FORCEINLINE void transfer_result(
        SourceState&& src, DestinationState const& dest)
    {
        hpx::detail::try_catch_exception_ptr(
            [&]() {
                traits::future_access<Future>::transfer_result(
                    traits::future_access<Future>::create(
                        HPX_FORWARD(SourceState, src)),
                    *dest);
            },
            [&](std::exception_ptr ep) {
                (*dest).set_exception(HPX_MOVE(ep));
            });
    }

    template <typename Func, typename Future, typename Continuation>
    void invoke_continuation_nounwrap(
        Func& func, Future&& future, Continuation& cont)
    {
        constexpr bool is_void =
            std::is_void_v<util::invoke_result_t<Func, Future&&>>;

        hpx::intrusive_ptr<Continuation> keep_alive(&cont);
        hpx::detail::try_catch_exception_ptr(
            [&]() {
                if constexpr (is_void)
                {
                    func(HPX_FORWARD(Future, future));
                    cont.set_value(util::unused);
                }
                else
                {
                    cont.set_value(func(HPX_FORWARD(Future, future)));
                }
            },
            [&](std::exception_ptr ep) { cont.set_exception(HPX_MOVE(ep)); });
    }

    template <typename Func, typename Future, typename Continuation>
    std::enable_if_t<!traits::detail::is_unique_future<
        util::invoke_result_t<Func, Future>>::value>
    invoke_continuation(Func& func, Future&& future, Continuation& cont)
    {
        hpx::scoped_annotation annotate(func);
        invoke_continuation_nounwrap(func, HPX_FORWARD(Future, future), cont);
    }

    template <typename Func, typename Future, typename Continuation>
    std::enable_if_t<traits::detail::is_unique_future<
        util::invoke_result_t<Func, Future>>::value>
    invoke_continuation(Func& func, Future&& future, Continuation& cont)
    {
        hpx::detail::try_catch_exception_ptr(
            [&]() {
                using inner_future = util::invoke_result_t<Func, Future>;
                using inner_shared_state_ptr =
                    traits::detail::shared_state_ptr_for_t<inner_future>;

                // take by value, as the future may go away immediately
                inner_shared_state_ptr inner_state =
                    traits::detail::get_shared_state(
                        func(HPX_FORWARD(Future, future)));
                typename inner_shared_state_ptr::element_type* ptr =
                    inner_state.get();

                if (ptr == nullptr)
                {
                    HPX_THROW_EXCEPTION(no_state, "invoke_continuation",
                        "the inner future has no valid shared state");
                }

                // Bind an on_completed handler to this future which will transfer
                // its result to the new future.
                hpx::intrusive_ptr<Continuation> cont_(&cont);
                ptr->execute_deferred();
                ptr->set_on_completed(
                    [inner_state = HPX_MOVE(inner_state),
                        cont_ = HPX_MOVE(cont_)]() mutable -> void {
                        return transfer_result<inner_future>(
                            HPX_MOVE(inner_state), HPX_MOVE(cont_));
                    });
            },
            [&](std::exception_ptr ep) { cont.set_exception(HPX_MOVE(ep)); });
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename Future, typename F, typename ContResult>
    class continuation : public detail::future_data<ContResult>
    {
    private:
        using base_type = future_data<ContResult>;

        using mutex_type = typename base_type::mutex_type;
        using result_type = typename base_type::result_type;

    protected:
        using base_type::mtx_;

        threads::thread_id_type get_id() const
        {
            std::lock_guard<mutex_type> l(mtx_);
            return id_;
        }
        void set_id(threads::thread_id_type const& id)
        {
            std::lock_guard<mutex_type> l(mtx_);
            id_ = id;
        }

        struct reset_id
        {
            reset_id(continuation& target)
              : target_(target)
            {
                if (threads::get_self_ptr() != nullptr)
                    target.set_id(threads::get_self_id());
            }
            ~reset_id()
            {
                target_.set_id(threads::invalid_thread_id);
            }
            continuation& target_;
        };

    public:
        using init_no_addref = typename base_type::init_no_addref;

        template <typename Func,
            typename Enable = std::enable_if_t<
                !std::is_same<std::decay_t<Func>, continuation>::value>>
        // NOLINTNEXTLINE(bugprone-forwarding-reference-overload)
        continuation(Func&& f)
          : started_(false)
          , id_(threads::invalid_thread_id)
          , f_(HPX_FORWARD(Func, f))
        {
        }

        template <typename Func>
        continuation(init_no_addref no_addref, Func&& f)
          : base_type(no_addref)
          , started_(false)
          , id_(threads::invalid_thread_id)
          , f_(HPX_FORWARD(Func, f))
        {
        }

    protected:
        void run_impl(traits::detail::shared_state_ptr_for_t<Future>&& f)
        {
            auto future = traits::future_access<std::decay_t<Future>>::create(
                HPX_MOVE(f));
            invoke_continuation(f_, HPX_MOVE(future), *this);
        }

        void run_impl_nounwrap(
            traits::detail::shared_state_ptr_for_t<Future>&& f)
        {
            auto future = traits::future_access<std::decay_t<Future>>::create(
                HPX_MOVE(f));
            invoke_continuation_nounwrap(f_, HPX_MOVE(future), *this);
        }

    public:
        void run(traits::detail::shared_state_ptr_for_t<Future>&& f,
            error_code& ec = throws)
        {
            {
                std::lock_guard<mutex_type> l(mtx_);
                if (started_)
                {
                    HPX_THROWS_IF(ec, task_already_started, "continuation::run",
                        "this task has already been started");
                    return;
                }
                started_ = true;
            }

            run_impl(HPX_MOVE(f));

            if (&ec != &throws)
                ec = make_success_code();
        }

        void run_nounwrap(traits::detail::shared_state_ptr_for_t<Future>&& f,
            error_code& ec = throws)
        {
            {
                std::lock_guard<mutex_type> l(mtx_);
                if (started_)
                {
                    HPX_THROWS_IF(ec, task_already_started,
                        "continuation::run_nounwrap",
                        "this task has already been started");
                    return;
                }
                started_ = true;
            }

            run_impl_nounwrap(HPX_MOVE(f));

            if (&ec != &throws)
                ec = make_success_code();
        }

    protected:
        void async_impl(traits::detail::shared_state_ptr_for_t<Future>&& f)
        {
            reset_id r(*this);

            auto future = traits::future_access<std::decay_t<Future>>::create(
                HPX_MOVE(f));
            invoke_continuation(f_, HPX_MOVE(future), *this);
        }

        void async_impl_nounwrap(
            traits::detail::shared_state_ptr_for_t<Future>&& f)
        {
            reset_id r(*this);

            auto future = traits::future_access<std::decay_t<Future>>::create(
                HPX_MOVE(f));
            invoke_continuation_nounwrap(f_, HPX_MOVE(future), *this);
        }

    public:
        ///////////////////////////////////////////////////////////////////////
        template <typename Spawner>
        void async(traits::detail::shared_state_ptr_for_t<Future>&& f,
            Spawner&& spawner, error_code& ec = hpx::throws)
        {
            {
                std::unique_lock<mutex_type> l(this->mtx_);
                if (started_)
                {
                    l.unlock();
                    HPX_THROWS_IF(ec, task_already_started,
                        "continuation::async",
                        "this task has already been started");
                    return;
                }
                started_ = true;
            }

            hpx::intrusive_ptr<continuation> this_(this);
            hpx::util::thread_description desc(f_, "async");
            spawner(
                [this_ = HPX_MOVE(this_), f = HPX_MOVE(f)]() mutable -> void {
                    this_->async_impl(HPX_MOVE(f));
                },
                desc);

            if (&ec != &throws)
                ec = make_success_code();
        }

        ///////////////////////////////////////////////////////////////////////
        template <typename Spawner>
        void async_nounwrap(traits::detail::shared_state_ptr_for_t<Future>&& f,
            Spawner&& spawner, error_code& ec = hpx::throws)
        {
            {
                std::unique_lock<mutex_type> l(this->mtx_);
                if (started_)
                {
                    l.unlock();
                    HPX_THROWS_IF(ec, task_already_started,
                        "continuation::async_nounwrap",
                        "this task has already been started");
                    return;
                }
                started_ = true;
            }

            hpx::intrusive_ptr<continuation> this_(this);
            hpx::util::thread_description desc(f_, "async_nounwrap");
            spawner(
                [this_ = HPX_MOVE(this_), f = HPX_MOVE(f)]() mutable -> void {
                    this_->async_impl_nounwrap(HPX_MOVE(f));
                },
                desc);

            if (&ec != &throws)
                ec = make_success_code();
        }

        ///////////////////////////////////////////////////////////////////////
        // cancellation support
        bool cancelable() const noexcept
        {
            return true;
        }

        void cancel()
        {
            std::unique_lock<mutex_type> l(this->mtx_);
            hpx::detail::try_catch_exception_ptr(
                [&]() {
                    if (!this->started_)
                        HPX_THROW_THREAD_INTERRUPTED_EXCEPTION();

                    if (this->is_ready())
                        return;    // nothing we can do

                    if (id_ != threads::invalid_thread_id)
                    {
                        // interrupt the executing thread
                        threads::interrupt_thread(id_);

                        this->started_ = true;

                        l.unlock();
                        this->set_error(future_cancelled,
                            "continuation<Future, ContResult>::cancel",
                            "future has been canceled");
                    }
                    else
                    {
                        l.unlock();
                        HPX_THROW_EXCEPTION(future_can_not_be_cancelled,
                            "continuation<Future, ContResult>::cancel",
                            "future can't be canceled at this time");
                    }
                },
                [&](std::exception_ptr ep) {
                    this->started_ = true;
                    this->set_exception(ep);
                    std::rethrow_exception(HPX_MOVE(ep));
                });
        }

    public:
        ///////////////////////////////////////////////////////////////////////
        // TODO: Reduce duplication!
        template <typename Spawner, typename Future_, typename Policy>
        void attach(Future_&& future, std::remove_reference_t<Spawner>& spawner,
            Policy&& policy, error_code& /*ec*/ = throws)
        {
            using shared_state_ptr =
                traits::detail::shared_state_ptr_for_t<Future_>;

            // bind an on_completed handler to this future which will invoke
            // the continuation
            hpx::intrusive_ptr<continuation> this_(this);
            shared_state_ptr state =
                traits::detail::get_shared_state(HPX_FORWARD(Future_, future));
            typename shared_state_ptr::element_type* ptr = state.get();

            if (ptr == nullptr)
            {
                HPX_THROW_EXCEPTION(no_state, "continuation::attach",
                    "the future to attach has no valid shared state");
            }

            ptr->execute_deferred();
            ptr->set_on_completed(
                [this_ = HPX_MOVE(this_), state = HPX_MOVE(state),
                    policy = HPX_FORWARD(Policy, policy),
                    &spawner]() mutable -> void {
                    if (hpx::detail::has_async_policy(policy))
                    {
                        this_->async(HPX_MOVE(state), spawner);
                    }
                    else
                    {
                        this_->run(HPX_MOVE(state));
                    }
                });
        }

        template <typename Spawner, typename Future_, typename Policy>
        void attach(Future_&& future,
            std::remove_reference_t<Spawner>&& spawner, Policy&& policy,
            error_code& /*ec*/ = throws)
        {
            using shared_state_ptr =
                traits::detail::shared_state_ptr_for_t<Future_>;

            // bind an on_completed handler to this future which will invoke
            // the continuation
            hpx::intrusive_ptr<continuation> this_(this);
            shared_state_ptr state =
                traits::detail::get_shared_state(HPX_FORWARD(Future_, future));
            typename shared_state_ptr::element_type* ptr = state.get();

            if (ptr == nullptr)
            {
                HPX_THROW_EXCEPTION(no_state, "continuation::attach",
                    "the future to attach has no valid shared state");
            }

            ptr->execute_deferred();
            ptr->set_on_completed(
                [this_ = HPX_MOVE(this_), state = HPX_MOVE(state),
                    policy = HPX_FORWARD(Policy, policy),
                    spawner = HPX_MOVE(spawner)]() mutable -> void {
                    if (hpx::detail::has_async_policy(policy))
                    {
                        this_->async(HPX_MOVE(state), HPX_MOVE(spawner));
                    }
                    else
                    {
                        this_->run(HPX_MOVE(state));
                    }
                });
        }

        ///////////////////////////////////////////////////////////////////////
        template <typename Spawner, typename Future_, typename Policy>
        void attach_nounwrap(Future_&& future,
            std::remove_reference_t<Spawner>& spawner, Policy&& policy,
            error_code& /*ec*/ = throws)
        {
            using shared_state_ptr =
                traits::detail::shared_state_ptr_for_t<Future_>;

            // bind an on_completed handler to this future which will invoke
            // the continuation
            hpx::intrusive_ptr<continuation> this_(this);
            shared_state_ptr state =
                traits::detail::get_shared_state(HPX_FORWARD(Future_, future));
            typename shared_state_ptr::element_type* ptr = state.get();

            if (ptr == nullptr)
            {
                HPX_THROW_EXCEPTION(no_state, "continuation::attach_nounwrap",
                    "the future to attach has no valid shared state");
            }

            ptr->execute_deferred();
            ptr->set_on_completed(
                [this_ = HPX_MOVE(this_), state = HPX_MOVE(state),
                    policy = HPX_FORWARD(Policy, policy),
                    &spawner]() mutable -> void {
                    if (hpx::detail::has_async_policy(policy))
                    {
                        this_->async_nounwrap(HPX_MOVE(state), spawner);
                    }
                    else
                    {
                        this_->run_nounwrap(HPX_MOVE(state));
                    }
                });
        }

        template <typename Spawner, typename Future_, typename Policy>
        void attach_nounwrap(Future_&& future,
            std::remove_reference_t<Spawner>&& spawner, Policy&& policy,
            error_code& /*ec*/ = throws)
        {
            using shared_state_ptr =
                traits::detail::shared_state_ptr_for_t<Future_>;

            // bind an on_completed handler to this future which will invoke
            // the continuation
            hpx::intrusive_ptr<continuation> this_(this);
            shared_state_ptr state =
                traits::detail::get_shared_state(HPX_FORWARD(Future_, future));
            typename shared_state_ptr::element_type* ptr = state.get();

            if (ptr == nullptr)
            {
                HPX_THROW_EXCEPTION(no_state, "continuation::attach_nounwrap",
                    "the future to attach has no valid shared state");
            }

            ptr->execute_deferred();
            ptr->set_on_completed(
                [this_ = HPX_MOVE(this_), state = HPX_MOVE(state),
                    policy = HPX_FORWARD(Policy, policy),
                    spawner = HPX_MOVE(spawner)]() mutable -> void {
                    if (hpx::detail::has_async_policy(policy))
                    {
                        this_->async_nounwrap(
                            HPX_MOVE(state), HPX_MOVE(spawner));
                    }
                    else
                    {
                        this_->run_nounwrap(HPX_MOVE(state));
                    }
                });
        }

    protected:
        bool started_;
        threads::thread_id_type id_;
        std::decay_t<F> f_;
    };

    template <typename Allocator, typename Future, typename F,
        typename ContResult>
    class continuation_allocator : public continuation<Future, F, ContResult>
    {
        using base_type = continuation<Future, F, ContResult>;

        using other_allocator = typename std::allocator_traits<
            Allocator>::template rebind_alloc<continuation_allocator>;

    public:
        using init_no_addref = typename base_type::init_no_addref;

        template <typename Func>
        continuation_allocator(other_allocator const& alloc, Func&& f)
          : base_type(HPX_FORWARD(Func, f))
          , alloc_(alloc)
        {
        }

        template <typename Func>
        continuation_allocator(
            init_no_addref no_addref, other_allocator const& alloc, Func&& f)
          : base_type(no_addref, HPX_FORWARD(Func, f))
          , alloc_(alloc)
        {
        }

    private:
        void destroy() noexcept override
        {
            using traits = std::allocator_traits<other_allocator>;

            other_allocator alloc(alloc_);
            traits::destroy(alloc, this);
            traits::deallocate(alloc, this, 1);
        }

        other_allocator alloc_;
    };
}}}    // namespace hpx::lcos::detail

namespace hpx { namespace traits { namespace detail {
    template <typename Future, typename F, typename ContResult,
        typename Allocator>
    struct shared_state_allocator<
        lcos::detail::continuation<Future, F, ContResult>, Allocator>
    {
        using type = lcos::detail::continuation_allocator<Allocator, Future, F,
            ContResult>;
    };
}}}    // namespace hpx::traits::detail

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace lcos { namespace detail {
    ///////////////////////////////////////////////////////////////////////////
    template <typename ContResult>
    class unwrap_continuation : public future_data<ContResult>
    {
    private:
        template <typename Inner>
        void on_inner_ready(
            traits::detail::shared_state_ptr_for_t<Inner>&& inner_state)
        {
            transfer_result<Inner>(HPX_MOVE(inner_state), this);
        }

        template <typename Outer>
        void on_outer_ready(
            traits::detail::shared_state_ptr_for_t<Outer>&& outer_state)
        {
            using inner_future = traits::future_traits_t<Outer>;
            using inner_shared_state_ptr =
                traits::detail::shared_state_ptr_for_t<inner_future>;

            // Bind an on_completed handler to this future which will transfer
            // its result to the new future.
            hpx::intrusive_ptr<unwrap_continuation> this_(this);

            hpx::detail::try_catch_exception_ptr(
                [&]() {
                    // if we get here, this future is ready
                    Outer outer = traits::future_access<Outer>::create(
                        HPX_MOVE(outer_state));

                    // take by value, as the future will go away immediately
                    inner_shared_state_ptr inner_state =
                        traits::detail::get_shared_state(outer.get());
                    typename inner_shared_state_ptr::element_type* ptr =
                        inner_state.get();

                    if (ptr == nullptr)
                    {
                        HPX_THROW_EXCEPTION(no_state,
                            "unwrap_continuation<ContResult>::on_outer_ready",
                            "the inner future has no valid shared state");
                    }

                    ptr->execute_deferred();
                    ptr->set_on_completed(
                        [this_ = HPX_MOVE(this_),
                            inner_state =
                                HPX_MOVE(inner_state)]() mutable -> void {
                            return this_->template on_inner_ready<inner_future>(
                                HPX_MOVE(inner_state));
                        });
                },
                [&](std::exception_ptr ep) {
                    this->set_exception(HPX_MOVE(ep));
                });
        }

    public:
        using init_no_addref = typename future_data<ContResult>::init_no_addref;

        unwrap_continuation() = default;

        unwrap_continuation(init_no_addref no_addref)
          : future_data<ContResult>(no_addref)
        {
        }

        template <typename Future>
        void attach(Future&& future)
        {
            using outer_shared_state_ptr =
                traits::detail::shared_state_ptr_for_t<Future>;

            // Bind an on_completed handler to this future which will wait for
            // the inner future and will transfer its result to the new future.
            hpx::intrusive_ptr<unwrap_continuation> this_(this);

            outer_shared_state_ptr outer_state =
                traits::detail::get_shared_state(future);
            typename outer_shared_state_ptr::element_type* ptr =
                outer_state.get();

            if (ptr == nullptr)
            {
                HPX_THROW_EXCEPTION(no_state,
                    "unwrap_continuation<ContResult>::attach",
                    "the future has no valid shared state");
            }

            ptr->execute_deferred();
            ptr->set_on_completed(
                [this_ = HPX_MOVE(this_),
                    outer_state = HPX_MOVE(outer_state)]() mutable -> void {
                    return this_->template on_outer_ready<Future>(
                        HPX_MOVE(outer_state));
                });
        }
    };

    template <typename Allocator, typename ContResult>
    class unwrap_continuation_allocator : public unwrap_continuation<ContResult>
    {
        using base_type = unwrap_continuation<ContResult>;

        using other_allocator = typename std::allocator_traits<
            Allocator>::template rebind_alloc<unwrap_continuation_allocator>;

    public:
        using init_no_addref = typename base_type::init_no_addref;

        unwrap_continuation_allocator(other_allocator const& alloc)
          : alloc_(alloc)
        {
        }

        unwrap_continuation_allocator(
            init_no_addref no_addref, other_allocator const& alloc)
          : base_type(no_addref)
          , alloc_(alloc)
        {
        }

    private:
        void destroy() noexcept override
        {
            using traits = std::allocator_traits<other_allocator>;

            other_allocator alloc(alloc_);
            traits::destroy(alloc, this);
            traits::deallocate(alloc, this, 1);
        }

        other_allocator alloc_;
    };
}}}    // namespace hpx::lcos::detail

namespace hpx { namespace traits { namespace detail {
    template <typename ContResult, typename Allocator>
    struct shared_state_allocator<lcos::detail::unwrap_continuation<ContResult>,
        Allocator>
    {
        using type =
            lcos::detail::unwrap_continuation_allocator<Allocator, ContResult>;
    };
}}}    // namespace hpx::traits::detail

namespace hpx { namespace lcos { namespace detail {
    template <typename Allocator, typename Future>
    inline traits::detail::shared_state_ptr_t<future_unwrap_result_t<Future>>
    unwrap_impl_alloc(Allocator const& a, Future&& future, error_code& /*ec*/)
    {
        using base_allocator = Allocator;
        using result_type = future_unwrap_result_t<Future>;

        using shared_state = traits::shared_state_allocator_t<
            detail::unwrap_continuation<result_type>, base_allocator>;

        using other_allocator = typename std::allocator_traits<
            base_allocator>::template rebind_alloc<shared_state>;
        using traits = std::allocator_traits<other_allocator>;

        using init_no_addref = typename shared_state::init_no_addref;

        using unique_ptr = std::unique_ptr<shared_state,
            util::allocator_deleter<other_allocator>>;

        other_allocator alloc(a);
        unique_ptr p(traits::allocate(alloc, 1),
            util::allocator_deleter<other_allocator>{alloc});

        traits::construct(alloc, p.get(), init_no_addref{}, alloc);

        // create a continuation
        hpx::traits::detail::shared_state_ptr_t<result_type> result(
            p.release(), false);
        static_cast<shared_state*>(result.get())
            ->attach(HPX_FORWARD(Future, future));
        return result;
    }

    template <typename Future>
    inline traits::detail::shared_state_ptr_t<future_unwrap_result_t<Future>>
    unwrap_impl(Future&& future, error_code& ec)
    {
        return unwrap_impl_alloc(
            util::internal_allocator<>{}, HPX_FORWARD(Future, future), ec);
    }

    template <typename Allocator, typename Future>
    inline traits::detail::shared_state_ptr_t<future_unwrap_result_t<Future>>
    unwrap_alloc(Allocator const& a, Future&& future, error_code& ec)
    {
        return unwrap_impl_alloc(a, HPX_FORWARD(Future, future), ec);
    }

    template <typename Future>
    inline traits::detail::shared_state_ptr_t<future_unwrap_result_t<Future>>
    unwrap(Future&& future, error_code& ec)
    {
        return unwrap_impl(HPX_FORWARD(Future, future), ec);
    }
}}}    // namespace hpx::lcos::detail
