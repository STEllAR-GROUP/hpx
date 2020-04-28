//  Copyright (c) 2007-2019 Hartmut Kaiser
//  Copyright (c) 2014-2015 Agustin Berge
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_LCOS_LOCAL_CONTINUATION_APR_17_2012_0150PM)
#define HPX_LCOS_LOCAL_CONTINUATION_APR_17_2012_0150PM

#include <hpx/config.hpp>
#include <hpx/allocator_support/allocator_deleter.hpp>
#include <hpx/allocator_support/internal_allocator.hpp>
#include <hpx/errors.hpp>
#include <hpx/lcos/detail/future_data.hpp>
#include <hpx/lcos/future_exec.hpp>
#include <hpx/memory/intrusive_ptr.hpp>
#include <hpx/runtime/launch_policy.hpp>
#include <hpx/traits/future_access.hpp>
#include <hpx/traits/future_traits.hpp>
#include <hpx/threading_base/annotated_function.hpp>
#include <hpx/threading_base/thread_description.hpp>

#include <hpx/execution/executors/execution.hpp>
#include <hpx/execution/detail/post_policy_dispatch.hpp>

#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <type_traits>
#include <utility>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace lcos { namespace detail
{
    template <typename Source, typename Destination>
    HPX_FORCEINLINE void transfer_result_impl(std::false_type,
        Source&& src, Destination& dest)
    {
        try {
            dest.set_value(src.get());
        }
        catch (...) {
            dest.set_exception(std::current_exception());
        }
    }

    template <typename Source, typename Destination>
    HPX_FORCEINLINE void transfer_result_impl(std::true_type,
        Source&& src, Destination& dest)
    {
        try {
            src.get();
            dest.set_value(util::unused);
        }
        catch (...) {
            dest.set_exception(std::current_exception());
        }
    }

    template <typename Future, typename SourceState, typename DestinationState>
    HPX_FORCEINLINE void transfer_result(
        SourceState&& src, DestinationState const& dest)
    {
        using is_void = std::is_void<
            typename traits::future_traits<Future>::type>;
        transfer_result_impl(is_void{},
            traits::future_access<Future>::create(std::forward<SourceState>(src)),
            *dest);
    }

    template <typename Func, typename Future, typename Continuation>
    void invoke_continuation_nounwrap(
        Func& func, Future&& future, Continuation& cont, std::false_type)
    {
        try {
            cont.set_value(func(std::forward<Future>(future)));
        }
        catch (...) {
            cont.set_exception(std::current_exception());
        }
    }

    template <typename Func, typename Future, typename Continuation>
    void invoke_continuation_nounwrap(
        Func& func, Future&& future, Continuation& cont, std::true_type)
    {
        try {
            func(std::forward<Future>(future));
            cont.set_value(util::unused);
        }
        catch (...) {
            cont.set_exception(std::current_exception());
        }
    }

    template <typename Func, typename Future, typename Continuation>
    typename std::enable_if<
       !traits::detail::is_unique_future<
            typename util::invoke_result<Func, Future>::type
        >::value
    >::type
    invoke_continuation(Func& func, Future && future, Continuation& cont)
    {
        typedef std::is_void<
            typename util::invoke_result<Func, Future>::type
        > is_void;

        hpx::util::annotate_function annotate(func);
        invoke_continuation_nounwrap(
            func, std::forward<Future>(future), cont, is_void());
    }

    template <typename Func, typename Future, typename Continuation>
    typename std::enable_if<
        traits::detail::is_unique_future<
            typename util::invoke_result<Func, Future>::type
        >::value
    >::type
    invoke_continuation(Func& func, Future && future, Continuation& cont)
    {
        try {
            typedef
                typename util::invoke_result<Func, Future>::type
                inner_future;
            typedef
                typename traits::detail::shared_state_ptr_for<inner_future>::type
                inner_shared_state_ptr;

            // take by value, as the future may go away immediately
            inner_shared_state_ptr inner_state =
                traits::detail::get_shared_state(func(std::forward<Future>(future)));
            typename inner_shared_state_ptr::element_type* ptr =
                inner_state.get();

            if (ptr == nullptr)
            {
                HPX_THROW_EXCEPTION(no_state,
                    "invoke_continuation",
                    "the inner future has no valid shared state");
            }

            // Bind an on_completed handler to this future which will transfer
            // its result to the new future.
            hpx::intrusive_ptr<Continuation> cont_(&cont);
            ptr->execute_deferred();
            ptr->set_on_completed(
                [inner_state = std::move(inner_state),
                    cont_ = std::move(cont_)
                ]() mutable -> void {
                    return transfer_result<inner_future>(
                        std::move(inner_state), std::move(cont_));
                });
        }
        catch (...) {
            cont.set_exception(std::current_exception());
        }
     }

    ///////////////////////////////////////////////////////////////////////////
    template <typename ContResult>
    struct continuation_result
    {
        typedef ContResult type;
    };

    template <typename ContResult>
    struct continuation_result<future<ContResult> >
    {
        typedef ContResult type;
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename Future, typename F, typename ContResult>
    class continuation : public detail::future_data<ContResult>
    {
    private:
        typedef future_data<ContResult> base_type;

        typedef typename base_type::mutex_type mutex_type;
        typedef typename base_type::result_type result_type;

    protected:
        threads::thread_id_type get_id() const
        {
            std::lock_guard<mutex_type> l(this->mtx_);
            return id_;
        }
        void set_id(threads::thread_id_type const& id)
        {
            std::lock_guard<mutex_type> l(this->mtx_);
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
        typedef typename base_type::init_no_addref init_no_addref;

        template <typename Func, typename Enable = typename
            std::enable_if<
                !std::is_same<typename std::decay<Func>::type,
                    continuation>::value>::type>
        // NOLINTNEXTLINE(bugprone-forwarding-reference-overload)
        continuation(Func && f)
          : started_(false), id_(threads::invalid_thread_id)
          , f_(std::forward<Func>(f))
        {}

        template <typename Func>
        continuation(init_no_addref no_addref, Func && f)
          : base_type(no_addref),
            started_(false), id_(threads::invalid_thread_id),
            f_(std::forward<Func>(f))
        {}

    protected:
        void run_impl(
            typename traits::detail::shared_state_ptr_for<
                Future
            >::type && f)
        {
            Future future = traits::future_access<Future>::create(std::move(f));
            invoke_continuation(f_, std::move(future), *this);
        }

        void run_impl_nounwrap(
            typename traits::detail::shared_state_ptr_for<
                Future
            >::type && f)
        {
            using is_void =
                std::is_void<typename util::invoke_result<F, Future>::type>;

            Future future = traits::future_access<Future>::create(std::move(f));
            invoke_continuation_nounwrap(
                f_, std::move(future), *this, is_void{});
        }

    public:
        void run(
            typename traits::detail::shared_state_ptr_for<
                Future
            >::type && f, threads::thread_priority, error_code& ec)
        {
            {
                std::lock_guard<mutex_type> l(this->mtx_);
                if (started_) {
                    HPX_THROWS_IF(ec, task_already_started,
                        "continuation::run",
                        "this task has already been started");
                    return;
                }
                started_ = true;
            }

            run_impl(std::move(f));

            if (&ec != &throws)
                ec = make_success_code();
        }

        void run(
            typename traits::detail::shared_state_ptr_for<
                Future
            >::type && f, threads::thread_priority priority)
        {
            run(std::move(f), priority, throws);
        }

        ///////////////////////////////////////////////////////////////////////
        void run_nounwrap(
            typename traits::detail::shared_state_ptr_for<
                Future
            >::type && f, threads::thread_priority, error_code& ec)
        {
            {
                std::lock_guard<mutex_type> l(this->mtx_);
                if (started_) {
                    HPX_THROWS_IF(ec, task_already_started,
                        "continuation::run_nounwrap",
                        "this task has already been started");
                    return;
                }
                started_ = true;
            }

            run_impl_nounwrap(std::move(f));

            if (&ec != &throws)
                ec = make_success_code();
        }

        void run_nounwrap(
            typename traits::detail::shared_state_ptr_for<
                Future
            >::type && f, threads::thread_priority priority)
        {
            run_nounwrap(std::move(f), priority, throws);
        }

    protected:
        void async_impl(
            typename traits::detail::shared_state_ptr_for<
                Future
            >::type && f)
        {
            reset_id r(*this);

            Future future = traits::future_access<Future>::create(std::move(f));
            invoke_continuation(f_, std::move(future), *this);
        }

        void async_impl_nounwrap(
            typename traits::detail::shared_state_ptr_for<
                Future
            >::type && f)
        {
            using is_void =
                std::is_void<typename util::invoke_result<F, Future>::type>;

            reset_id r(*this);

            Future future = traits::future_access<Future>::create(std::move(f));
            invoke_continuation_nounwrap(
                f_, std::move(future), *this, is_void{});
        }

        void async_exec_impl(
            typename traits::detail::shared_state_ptr_for<
                Future
            >::type && f)
        {
            using is_void =
                std::is_void<typename util::invoke_result<F, Future>::type>;

            reset_id r(*this);

            Future future = traits::future_access<Future>::create(std::move(f));
            invoke_continuation_nounwrap(
                f_, std::move(future), *this, is_void{});
        }

    public:
        ///////////////////////////////////////////////////////////////////////
        void async(
            typename traits::detail::shared_state_ptr_for<
                Future
            >::type && f,
            threads::thread_priority /*priority*/,
            error_code& ec)
        {
            {
                std::unique_lock<mutex_type> l(this->mtx_);
                if (started_) {
                    l.unlock();
                    HPX_THROWS_IF(ec, task_already_started,
                        "continuation::async",
                        "this task has already been started");
                    return;
                }
                started_ = true;
            }

            hpx::intrusive_ptr<continuation> this_(this);
            hpx::util::thread_description desc(f_,
                "hpx::parallel::execution::parallel_executor::post");

            parallel::execution::detail::post_policy_dispatch<
                    hpx::launch::async_policy
                >::call(hpx::launch::async, desc,
                    [this_ = std::move(this_),
                        f = std::move(f)
                    ]() mutable -> void {
                        this_->async_impl(std::move(f));
                    });

            if (&ec != &throws)
                ec = make_success_code();
        }

        void async(
            typename traits::detail::shared_state_ptr_for<
                Future
            >::type && f,
            threads::thread_priority priority)
        {
            async(std::move(f), priority, throws);
        }

        ///////////////////////////////////////////////////////////////////////
        void async_nounwrap(
            typename traits::detail::shared_state_ptr_for<
                Future
            >::type && f,
            threads::thread_priority /*priority*/,
            error_code& ec)
        {
            {
                std::unique_lock<mutex_type> l(this->mtx_);
                if (started_) {
                    l.unlock();
                    HPX_THROWS_IF(ec, task_already_started,
                        "continuation::async",
                        "this task has already been started");
                    return;
                }
                started_ = true;
            }

            hpx::intrusive_ptr<continuation> this_(this);
            hpx::util::thread_description desc(f_,
                "hpx::parallel::execution::parallel_executor::post");

            parallel::execution::detail::post_policy_dispatch<
                    hpx::launch::async_policy
                >::call(hpx::launch::async, desc,
                    [this_ = std::move(this_),
                        f = std::move(f)
                    ]() mutable -> void {
                        this_->async_impl_nounwrap(std::move(f));
                    });

            if (&ec != &throws)
                ec = make_success_code();
        }

        void async_nounwrap(
            typename traits::detail::shared_state_ptr_for<
                Future
            >::type && f,
            threads::thread_priority priority)
        {
            async_nounwrap(std::move(f), priority, throws);
        }

        ///////////////////////////////////////////////////////////////////////
        template <typename Executor>
        void async_exec(
            typename traits::detail::shared_state_ptr_for<
                Future
            >::type && f,
            Executor && exec, error_code& ec)
        {
            {
                std::unique_lock<mutex_type> l(this->mtx_);
                if (started_) {
                    l.unlock();
                    HPX_THROWS_IF(ec, task_already_started,
                        "continuation::async_exec",
                        "this task has already been started");
                    return;
                }
                started_ = true;
            }

            hpx::intrusive_ptr<continuation> this_(this);
            parallel::execution::post(std::forward<Executor>(exec),
                [this_ = std::move(this_),
                    f = std::move(f)
                ]() mutable -> void {
                    this_->async_exec_impl(std::move(f));
                });

            if (&ec != &throws)
                ec = make_success_code();
        }

        template <typename Executor>
        void async_exec(
            typename traits::detail::shared_state_ptr_for<
                Future
            >::type && f,
            Executor && exec)
        {
            async_exec(std::move(f), std::forward<Executor>(exec), throws);
        }

        ///////////////////////////////////////////////////////////////////////
        // cancellation support
        bool cancelable() const
        {
            return true;
        }

        void cancel()
        {
            std::unique_lock<mutex_type> l(this->mtx_);
            try {
                if (!this->started_)
                    HPX_THROW_THREAD_INTERRUPTED_EXCEPTION();

                if (this->is_ready())
                    return;   // nothing we can do

                if (id_ != threads::invalid_thread_id) {
                    // interrupt the executing thread
                    threads::interrupt_thread(id_);

                    this->started_ = true;

                    l.unlock();
                    this->set_error(future_cancelled,
                        "continuation<Future, ContResult>::cancel",
                        "future has been canceled");
                }
                else {
                    l.unlock();
                    HPX_THROW_EXCEPTION(future_can_not_be_cancelled,
                        "continuation<Future, ContResult>::cancel",
                        "future can't be canceled at this time");
                }
            }
            catch (...) {
                this->started_ = true;
                this->set_exception(std::current_exception());
                throw;
            }
        }

    public:
        ///////////////////////////////////////////////////////////////////////
        template <typename Policy>
        void attach(Future const& future, Policy && policy)
        {
            typedef
                typename traits::detail::shared_state_ptr_for<Future>::type
                shared_state_ptr;

            // bind an on_completed handler to this future which will invoke
            // the continuation
            hpx::intrusive_ptr<continuation> this_(this);

            shared_state_ptr state = traits::detail::get_shared_state(future);
            typename shared_state_ptr::element_type* ptr = state.get();

            if (ptr == nullptr)
            {
                HPX_THROW_EXCEPTION(no_state,
                    "continuation::attach",
                    "the future to attach has no valid shared state");
            }

            ptr->execute_deferred();
            ptr->set_on_completed(
                [this_ = std::move(this_),
                    state = std::move(state),
                    policy = std::forward<Policy>(policy)
                ]() mutable -> void {
                    if (hpx::detail::has_async_policy(policy))
                        this_->async(std::move(state), policy.priority());
                    else
                        this_->run(std::move(state), policy.priority());
                });
        }

        ///////////////////////////////////////////////////////////////////////
        template <typename Policy>
        void attach_nounwrap(Future const& future, Policy && policy)
        {
            typedef
                typename traits::detail::shared_state_ptr_for<Future>::type
                shared_state_ptr;

            // bind an on_completed handler to this future which will invoke
            // the continuation
            hpx::intrusive_ptr<continuation> this_(this);

            shared_state_ptr state = traits::detail::get_shared_state(future);
            typename shared_state_ptr::element_type* ptr = state.get();

            if (ptr == nullptr)
            {
                HPX_THROW_EXCEPTION(no_state,
                    "continuation::attach_nounwrap",
                    "the future to attach has no valid shared state");
            }

            ptr->execute_deferred();
            ptr->set_on_completed(
                [this_ = std::move(this_),
                    state = std::move(state),
                    policy = std::forward<Policy>(policy)
                ]() mutable -> void {
                    if (hpx::detail::has_async_policy(policy))
                        this_->async_nounwrap(std::move(state), policy.priority());
                    else
                        this_->run_nounwrap(std::move(state), policy.priority());
                });
        }

        ///////////////////////////////////////////////////////////////////////
        template <typename Executor>
        void attach_exec(Future const& future,
            typename std::remove_reference<Executor>::type& exec)
        {
            typedef
                typename traits::detail::shared_state_ptr_for<Future>::type
                shared_state_ptr;

            // bind an on_completed handler to this future which will invoke
            // the continuation
            hpx::intrusive_ptr<continuation> this_(this);
            shared_state_ptr state = traits::detail::get_shared_state(future);
            typename shared_state_ptr::element_type* ptr = state.get();

            if (ptr == nullptr)
            {
                HPX_THROW_EXCEPTION(no_state,
                    "continuation::attach_exec",
                    "the future to attach has no valid shared state");
            }

            ptr->execute_deferred();
            ptr->set_on_completed(
                [this_ = std::move(this_),
                    state = std::move(state),
                    &exec
                ]() mutable -> void {
                    this_->async_exec(std::move(state), exec);
                });
        }

        template <typename Executor>
        void attach_exec(Future const& future,
            typename std::remove_reference<Executor>::type&& exec)
        {
            typedef
                typename traits::detail::shared_state_ptr_for<Future>::type
                shared_state_ptr;

            // bind an on_completed handler to this future which will invoke
            // the continuation
            hpx::intrusive_ptr<continuation> this_(this);
            shared_state_ptr state = traits::detail::get_shared_state(future);
            typename shared_state_ptr::element_type* ptr = state.get();

            if (ptr == nullptr)
            {
                HPX_THROW_EXCEPTION(no_state,
                    "continuation::attach_exec",
                    "the future to attach has no valid shared state");
            }

            ptr->execute_deferred();
            ptr->set_on_completed(
                [this_ = std::move(this_),
                    state = std::move(state),
                    exec = std::move(exec)
                ]() mutable -> void {
                    return this_->async_exec(std::move(state), std::move(exec));
                });
        }

    protected:
        bool started_;
        threads::thread_id_type id_;
        typename std::decay<F>::type f_;
    };

    template <typename Allocator, typename Future, typename F,
        typename ContResult>
    class continuation_allocator : public continuation<Future, F, ContResult>
    {
        typedef continuation<Future, F, ContResult> base_type;

        typedef typename
                std::allocator_traits<Allocator>::template
                    rebind_alloc<continuation_allocator>
            other_allocator;

    public:
        typedef typename base_type::init_no_addref init_no_addref;

        template <typename Func>
        continuation_allocator(other_allocator const& alloc, Func && f)
          : base_type(std::forward<Func>(f))
          , alloc_(alloc)
        {}

        template <typename Func>
        continuation_allocator(init_no_addref no_addref,
                other_allocator const& alloc, Func && f)
          : base_type(no_addref, std::forward<Func>(f))
          , alloc_(alloc)
        {}

    private:
        void destroy() override
        {
            typedef std::allocator_traits<other_allocator> traits;

            other_allocator alloc(alloc_);
            traits::destroy(alloc, this);
            traits::deallocate(alloc, this, 1);
        }

        other_allocator alloc_;
    };
}}}

namespace hpx { namespace traits { namespace detail
{
    template <typename Future, typename F, typename ContResult,
        typename Allocator>
    struct shared_state_allocator<
        lcos::detail::continuation<Future, F, ContResult>, Allocator>
    {
        typedef lcos::detail::continuation_allocator<
            Allocator, Future, F, ContResult> type;
    };
}}}

namespace hpx { namespace lcos { namespace detail
{
    ///////////////////////////////////////////////////////////////////////////
    template <typename ContResult, typename Future, typename Policy, typename F>
    inline typename traits::detail::shared_state_ptr<
        typename continuation_result<ContResult>::type
    >::type
    make_continuation(Future const& future, Policy && policy, F && f)
    {
        typedef typename continuation_result<ContResult>::type result_type;
        typedef detail::continuation<Future, F, result_type> shared_state;
        typedef typename shared_state::init_no_addref init_no_addref;

        // create a continuation
        typename traits::detail::shared_state_ptr<result_type>::type p(
            new shared_state(init_no_addref{}, std::forward<F>(f)), false);
        static_cast<shared_state*>(p.get())->attach(
            future, std::forward<Policy>(policy));
        return p;
    }

    // same as above, except with allocator
    template <typename ContResult, typename Allocator, typename Future,
        typename Policy, typename F>
    inline typename traits::detail::shared_state_ptr<
        typename continuation_result<ContResult>::type
    >::type
    make_continuation_alloc(Allocator const& a,
        Future const& future, Policy&& policy, F&& f)
    {
        using result_type = typename continuation_result<ContResult>::type;

        using base_allocator = Allocator;
        using shared_state = typename traits::detail::shared_state_allocator<
                detail::continuation<Future, F, result_type>, base_allocator
            >::type;

        using other_allocator = typename std::allocator_traits<base_allocator>::
            template rebind_alloc<shared_state>;
        using traits = std::allocator_traits<other_allocator>;

        using init_no_addref = typename shared_state::init_no_addref;

        using unique_ptr = std::unique_ptr<shared_state,
            util::allocator_deleter<other_allocator>>;

        other_allocator alloc(a);
        unique_ptr p(traits::allocate(alloc, 1),
            util::allocator_deleter<other_allocator>{alloc});
        traits::construct(
            alloc, p.get(), init_no_addref{}, alloc, std::forward<F>(f));

        // create a continuation
        typename hpx::traits::detail::shared_state_ptr<result_type>::type r(
            p.release(), false);

        static_cast<shared_state*>(r.get())->attach(
            future, std::forward<Policy>(policy));

        return r;
    }

    // same as above, except with allocator and without unwrapping returned
    // futures
    template <typename ContResult, typename Allocator, typename Future,
        typename Policy, typename F>
    inline typename traits::detail::shared_state_ptr<ContResult>::type
    make_continuation_alloc_nounwrap(Allocator const& a,
        Future const& future, Policy&& policy, F&& f)
    {
        using result_type = ContResult;

        using base_allocator = Allocator;
        using shared_state = typename traits::detail::shared_state_allocator<
                detail::continuation<Future, F, result_type>, base_allocator
            >::type;

        using other_allocator = typename std::allocator_traits<base_allocator>::
            template rebind_alloc<shared_state>;
        using traits = std::allocator_traits<other_allocator>;

        using init_no_addref = typename shared_state::init_no_addref;

        using unique_ptr = std::unique_ptr<shared_state,
            util::allocator_deleter<other_allocator>>;

        other_allocator alloc(a);
        unique_ptr p(traits::allocate(alloc, 1),
            util::allocator_deleter<other_allocator>{alloc});
        traits::construct(
            alloc, p.get(), init_no_addref{}, alloc, std::forward<F>(f));

        // create a continuation
        typename hpx::traits::detail::shared_state_ptr<result_type>::type r(
            p.release(), false);

        static_cast<shared_state*>(r.get())->attach_nounwrap(
            future, std::forward<Policy>(policy));

        return r;
    }

    template <typename ContResult, typename Future, typename Executor,
        typename F>
    inline typename traits::detail::shared_state_ptr<ContResult>::type
    make_continuation_exec(Future const& future, Executor && exec, F && f)
    {
        typedef detail::continuation<Future, F, ContResult> shared_state;
        typedef typename shared_state::init_no_addref init_no_addref;

        // create a continuation
        typename traits::detail::shared_state_ptr<ContResult>::type p(
            new shared_state(init_no_addref{}, std::forward<F>(f)), false);
        static_cast<shared_state*>(p.get())->template attach_exec<Executor>(
            future, exec);
        return p;
    }

    template <typename Executor, typename Future, typename F>
    inline typename hpx::traits::future_then_executor_result<
        Executor, typename std::decay<Future>::type, F
    >::type
    then_execute_helper(Executor && exec, F && f, Future && predecessor)
    {
        // simply forward this to executor
        return parallel::execution::then_execute(exec, std::forward<F>(f),
            std::forward<Future>(predecessor));
    }
}}}

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace lcos { namespace detail
{
    ///////////////////////////////////////////////////////////////////////////
    template <typename ContResult>
    class unwrap_continuation : public future_data<ContResult>
    {
    private:
        template <typename Inner>
        void on_inner_ready(
            typename traits::detail::shared_state_ptr_for<
                Inner
            >::type && inner_state)
        {
            try {
                transfer_result<Inner>(std::move(inner_state), this);
            }
            catch (...) {
                this->set_exception(std::current_exception());
            }
        }

        template <typename Outer>
        void on_outer_ready(
            typename traits::detail::shared_state_ptr_for<
                Outer
            >::type && outer_state)
        {
            typedef typename traits::future_traits<Outer>::type inner_future;
            typedef
                typename traits::detail::shared_state_ptr_for<inner_future>::type
                inner_shared_state_ptr;

            // Bind an on_completed handler to this future which will transfer
            // its result to the new future.
            hpx::intrusive_ptr<unwrap_continuation> this_(this);
            try {
                // if we get here, this future is ready
                Outer outer = traits::future_access<Outer>::create(
                    std::move(outer_state));

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
                    [this_ = std::move(this_),
                        inner_state = std::move(inner_state)
                    ]() mutable -> void {
                        return this_->template on_inner_ready<inner_future>(
                            std::move(inner_state));
                    });
            }
            catch (...) {
                this->set_exception(std::current_exception());
            }
        }

    public:
        typedef typename future_data<ContResult>::init_no_addref init_no_addref;

        unwrap_continuation() {}

        unwrap_continuation(init_no_addref no_addref)
          : future_data<ContResult>(no_addref)
        {}

        template <typename Future>
        void attach(Future && future)
        {
            typedef
                typename traits::detail::shared_state_ptr_for<Future>::type
                outer_shared_state_ptr;

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
                [this_ = std::move(this_),
                    outer_state = std::move(outer_state)
                ]() mutable -> void {
                    return this_->template on_outer_ready<Future>(std::move(outer_state));
                });
        }
    };

    template <typename Allocator, typename ContResult>
    class unwrap_continuation_allocator
      : public unwrap_continuation<ContResult>
    {
        using base_type = unwrap_continuation<ContResult>;

        using other_allocator = typename std::allocator_traits<
            Allocator>::template rebind_alloc<unwrap_continuation_allocator>;

    public:
        using init_no_addref = typename base_type::init_no_addref;

        unwrap_continuation_allocator(other_allocator const& alloc)
          : alloc_(alloc)
        {}

        unwrap_continuation_allocator(
                init_no_addref no_addref, other_allocator const& alloc)
          : base_type(no_addref)
          , alloc_(alloc)
        {}

    private:
        void destroy() override
        {
            typedef std::allocator_traits<other_allocator> traits;

            other_allocator alloc(alloc_);
            traits::destroy(alloc, this);
            traits::deallocate(alloc, this, 1);
        }

        other_allocator alloc_;
    };
}}}

namespace hpx { namespace traits { namespace detail
{
    template <typename ContResult, typename Allocator>
    struct shared_state_allocator<
        lcos::detail::unwrap_continuation<ContResult>, Allocator>
    {
        using type =
            lcos::detail::unwrap_continuation_allocator<Allocator, ContResult>;
    };
}}}

namespace hpx { namespace lcos { namespace detail
{
    template <typename Allocator, typename Future>
    inline typename traits::detail::shared_state_ptr<
        typename future_unwrap_result<Future>::result_type>::type
    unwrap_impl_alloc(Allocator const& a, Future && future, error_code& /*ec*/)
    {
        using base_allocator = Allocator;
        using result_type = typename future_unwrap_result<Future>::result_type;

        using shared_state = typename traits::detail::shared_state_allocator<
                detail::unwrap_continuation<result_type>, base_allocator
            >::type;

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
        typename hpx::traits::detail::shared_state_ptr<result_type>::type
            result(p.release(), false);
        static_cast<shared_state*>(result.get())
            ->attach(std::forward<Future>(future));
        return result;
    }

    template <typename Future>
    inline typename traits::detail::shared_state_ptr<
        typename future_unwrap_result<Future>::result_type>::type
    unwrap_impl(Future && future, error_code& ec)
    {
        return unwrap_impl_alloc(
            util::internal_allocator<>{}, std::forward<Future>(future), ec);
    }

    template <typename Allocator, typename Future>
    inline typename traits::detail::shared_state_ptr<
        typename future_unwrap_result<Future>::result_type>::type
    unwrap_alloc(Allocator const& a, Future && future, error_code& ec)
    {
        return unwrap_impl_alloc(a, std::forward<Future>(future), ec);
    }

    template <typename Future>
    inline typename traits::detail::shared_state_ptr<
        typename future_unwrap_result<Future>::result_type>::type
    unwrap(Future && future, error_code& ec)
    {
        return unwrap_impl(std::forward<Future>(future), ec);
    }
}}}

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace lcos { namespace detail
{
    template <typename Future>
    inline typename traits::detail::shared_state_ptr<void>::type
    downcast_to_void(Future& future, bool addref)
    {
        typedef typename traits::detail::shared_state_ptr<void>::type
            shared_state_type;
        typedef typename shared_state_type::element_type element_type;

        // same as static_pointer_cast, but with addref option
        return shared_state_type(static_cast<element_type*>(
                traits::detail::get_shared_state(future).get()
            ), addref);
    }
}}}

#endif
