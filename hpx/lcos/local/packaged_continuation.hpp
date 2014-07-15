//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_LCOS_LOCAL_CONTINUATION_APR_17_2012_0150PM)
#define HPX_LCOS_LOCAL_CONTINUATION_APR_17_2012_0150PM

#include <hpx/hpx_fwd.hpp>
#include <hpx/traits/promise_remote_result.hpp>
#include <hpx/traits/is_future.hpp>
#include <hpx/util/decay.hpp>
#include <hpx/util/move.hpp>
#include <hpx/lcos/detail/future_data.hpp>
#include <hpx/lcos/future.hpp>

#include <boost/shared_ptr.hpp>
#include <boost/make_shared.hpp>
#include <boost/intrusive_ptr.hpp>
#include <boost/type_traits/remove_reference.hpp>
#include <utility>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace lcos { namespace detail
{
    template <typename Future>
    struct transfer_result
    {
        template <typename Source, typename Destination>
        void apply(Source&& src, Destination& dest, boost::mpl::false_) const
        {
            try {
                dest.set_result(src.get());
            }
            catch (...) {
                dest.set_exception(boost::current_exception());
            }
        }

        template <typename Source, typename Destination>
        void apply(Source&& src, Destination& dest, boost::mpl::true_) const
        {
            try {
                src.get();
                dest.set_result(util::unused);
            }
            catch (...) {
                dest.set_exception(boost::current_exception());
            }
        }

        template <typename SourceState, typename DestinationState>
        void operator()(SourceState& src, DestinationState const& dest) const
        {
            typedef typename boost::is_void<
                typename traits::future_traits<Future>::type
            >::type is_void;

            apply(traits::future_access<Future>::create(src), *dest, is_void());
        }
    };

    template <typename Func, typename Future, typename Continuation>
    void invoke_continuation(Func& func, Future& future, Continuation& cont,
        boost::mpl::false_)
    {
        try {
            cont.set_result(func(std::move(future)));
        }
        catch (...) {
            cont.set_exception(boost::current_exception());
        }
    }

    template <typename Func, typename Future, typename Continuation>
    void invoke_continuation(Func& func, Future& future, Continuation& cont,
        boost::mpl::true_)
    {
        try {
            func(std::move(future));
            cont.set_result(util::unused);
        }
        catch (...) {
            cont.set_exception(boost::current_exception());
        }
    }

    template <typename Func, typename Future, typename Continuation>
    typename boost::disable_if<
        traits::detail::is_unique_future<typename util::result_of<Func(Future)>::type>
    >::type invoke_continuation(Func& func, Future& future, Continuation& cont)
    {
        typedef typename boost::is_void<
            typename util::result_of<Func(Future)>::type
        >::type is_void;

        invoke_continuation(func, future, cont, is_void());
    }

    template <typename Func, typename Future, typename Continuation>
    typename boost::enable_if<
        traits::detail::is_unique_future<typename util::result_of<Func(Future)>::type>
    >::type invoke_continuation(Func& func, Future& future, Continuation& cont)
    {
        try {
            typedef
                typename util::result_of<Func(Future)>::type
                inner_future;
            typedef
                typename shared_state_ptr_for<inner_future>::type
                inner_shared_state_ptr;

            // take by value, as the future may go away immediately
            inner_shared_state_ptr inner_state =
                lcos::detail::get_shared_state(func(std::move(future)));

            if (inner_state.get() == 0)
            {
                HPX_THROW_EXCEPTION(no_state,
                    "invoke_continuation",
                    "the inner future has no valid shared state");
            }

            // Bind an on_completed handler to this future which will transfer
            // its result to the new future.
            boost::intrusive_ptr<Continuation> cont_(&cont);
            inner_state->set_on_completed(util::bind(
                transfer_result<inner_future>(), inner_state, cont_));
        }
        catch (...) {
            cont.set_exception(boost::current_exception());
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
    class continuation
      : public future_data<typename continuation_result<ContResult>::type>
    {
    private:
        typedef future_data<ContResult> base_type;

        typedef typename base_type::mutex_type mutex_type;
        typedef typename base_type::result_type result_type;

    protected:
        threads::thread_id_type get_id() const
        {
            typename mutex_type::scoped_lock l(this->mtx_);
            return id_;
        }
        void set_id(threads::thread_id_type const& id)
        {
            typename mutex_type::scoped_lock l(this->mtx_);
            id_ = id;
        }

        struct reset_id
        {
            reset_id(continuation& target)
              : target_(target)
            {
                if (threads::get_self_ptr() != 0)
                    target.set_id(threads::get_self_id());
            }
            ~reset_id()
            {
                target_.set_id(threads::invalid_thread_id);
            }
            continuation& target_;
        };

    public:
        template <typename Func>
        continuation(Func && f)
          : started_(false), id_(threads::invalid_thread_id)
          , f_(std::forward<Func>(f))
        {}

        void run_impl(typename shared_state_ptr_for<Future>::type const& f)
        {
            Future future = traits::future_access<Future>::create(f);
            invoke_continuation(f_, future, *this);
        }

        void run(typename shared_state_ptr_for<Future>::type const& f,
            error_code& ec)
        {
            {
                typename mutex_type::scoped_lock l(this->mtx_);
                if (started_) {
                    HPX_THROWS_IF(ec, task_already_started,
                        "continuation::run",
                        "this task has already been started");
                    return;
                }
                started_ = true;
            }

            run_impl(f);

            if (&ec != &throws)
                ec = make_success_code();
        }

        void run(typename shared_state_ptr_for<Future>::type const& f)
        {
            run(f, throws);
        }

        threads::thread_state_enum
        async_impl(typename shared_state_ptr_for<Future>::type const& f)
        {
            reset_id r(*this);

            Future future = traits::future_access<Future>::create(f);
            invoke_continuation(f_, future, *this);
            return threads::terminated;
        }

        void async(typename shared_state_ptr_for<Future>::type const& f,
            error_code& ec)
        {
            {
                typename mutex_type::scoped_lock l(this->mtx_);
                if (started_) {
                    HPX_THROWS_IF(ec, task_already_started,
                        "continuation::async",
                        "this task has already been started");
                    return;
                }
                started_ = true;
            }

            boost::intrusive_ptr<continuation> this_(this);
            threads::thread_state_enum (continuation::*async_impl_ptr)(
                typename shared_state_ptr_for<Future>::type const&
            ) = &continuation::async_impl;

            applier::register_thread_plain(
                util::bind(async_impl_ptr, std::move(this_), f),
                "continuation::async");

            if (&ec != &throws)
                ec = make_success_code();
        }

        void async(typename shared_state_ptr_for<Future>::type const& f,
            threads::executor& sched, error_code& ec)
        {
            {
                typename mutex_type::scoped_lock l(this->mtx_);
                if (started_) {
                    HPX_THROWS_IF(ec, task_already_started,
                        "continuation::async",
                        "this task has already been started");
                    return;
                }
                started_ = true;
            }

            boost::intrusive_ptr<continuation> this_(this);
            threads::thread_state_enum (continuation::*async_impl_ptr)(
                typename shared_state_ptr_for<Future>::type const&
            ) = &continuation::async_impl;

            sched.add(
                util::bind(async_impl_ptr, std::move(this_), f),
                "continuation::async");

            if (&ec != &throws)
                ec = make_success_code();
        }

        void async(typename shared_state_ptr_for<Future>::type const& f)
        {
            async(f, throws);
        }

        void async(typename shared_state_ptr_for<Future>::type const& f,
            threads::executor& sched)
        {
            async(f, sched, throws);
        }

        void deleting_owner()
        {
            typename mutex_type::scoped_lock l(this->mtx_);
            if (!started_) {
                started_ = true;
                l.unlock();
                this->set_error(broken_task,
                    "continuation::deleting_owner",
                    "deleting task owner before future has been executed");
            }
        }

        // cancellation support
        bool cancelable() const
        {
            return true;
        }

        void cancel()
        {
            typename mutex_type::scoped_lock l(this->mtx_);
            try {
                if (!this->started_)
                    boost::throw_exception(hpx::thread_interrupted());

                if (this->is_ready_locked())
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
                    HPX_THROW_EXCEPTION(future_can_not_be_cancelled,
                        "continuation<Future, ContResult>::cancel",
                        "future can't be canceled at this time");
                }
            }
            catch (hpx::exception const&) {
                this->started_ = true;
                this->set_exception(boost::current_exception());
                throw;
            }
        }

    public:
        void attach(Future& future, BOOST_SCOPED_ENUM(launch) policy)
        {
            typedef
                typename shared_state_ptr_for<Future>::type
                shared_state_ptr;

            // bind an on_completed handler to this future which will invoke
            // the continuation
            boost::intrusive_ptr<continuation> this_(this);
            void (continuation::*cb)(shared_state_ptr const&);
            if (policy & launch::sync)
                cb = &continuation::run;
            else
                cb = &continuation::async;

            if (future.wait_for(boost::posix_time::seconds(0)) == future_status::deferred)
                future.wait();

            shared_state_ptr const& state =
                lcos::detail::get_shared_state(future);
            state->set_on_completed(util::bind(cb, std::move(this_), state));
        }

        void attach(Future& future, threads::executor& sched)
        {
            typedef
                typename shared_state_ptr_for<Future>::type
                shared_state_ptr;

            // bind an on_completed handler to this future which will invoke
            // the continuation
            boost::intrusive_ptr<continuation> this_(this);
            void (continuation::*cb)(shared_state_ptr const&, threads::executor&) =
                &continuation::async;

            if (future.wait_for(boost::posix_time::seconds(0)) == future_status::deferred)
                future.wait();

            shared_state_ptr const& state =
                lcos::detail::get_shared_state(future);
            state->set_on_completed(util::bind(cb, std::move(this_), state, boost::ref(sched)));
        }

    protected:
        bool started_;
        threads::thread_id_type id_;
        typename util::decay<F>::type f_;
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename ContResult, typename Future, typename F>
    inline typename shared_state_ptr<
        typename continuation_result<ContResult>::type
    >::type
    make_continuation(Future& future, BOOST_SCOPED_ENUM(launch) policy,
        F && f)
    {
        typedef detail::continuation<Future, F, ContResult> shared_state;
        typedef typename continuation_result<ContResult>::type result_type;

        // create a continuation
        typename shared_state_ptr<result_type>::type p(
            new shared_state(std::forward<F>(f)));
        static_cast<shared_state*>(p.get())->attach(future, policy);
        return std::move(p);
    }

    template <typename ContResult, typename Future, typename F>
    inline typename shared_state_ptr<
        typename continuation_result<ContResult>::type
    >::type
    make_continuation(Future& future, threads::executor& sched,
        F && f)
    {
        typedef detail::continuation<Future, F, ContResult> shared_state;
        typedef typename continuation_result<ContResult>::type result_type;

        // create a continuation
        typename shared_state_ptr<result_type>::type p(
            new shared_state(std::forward<F>(f)));
        static_cast<shared_state*>(p.get())->attach(future, sched);
        return std::move(p);
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
            typename shared_state_ptr_for<Inner>::type const& inner_state)
        {
            try {
                unwrap_continuation* this_ = this;
                transfer_result<Inner>()(inner_state, this_);
            }
            catch (...) {
                this->set_exception(boost::current_exception());
            }
        }

        template <typename Outer>
        void on_outer_ready(
            typename shared_state_ptr_for<Outer>::type const& outer_state)
        {
            typedef typename traits::future_traits<Outer>::type inner_future;
            typedef
                typename shared_state_ptr_for<inner_future>::type
                inner_shared_state_ptr;

            // Bind an on_completed handler to this future which will transfer
            // its result to the new future.
            boost::intrusive_ptr<unwrap_continuation> this_(this);
            void (unwrap_continuation::*inner_ready)(
                inner_shared_state_ptr const&) =
                    &unwrap_continuation::on_inner_ready<inner_future>;

            try {
                // if we get here, this future is ready
                Outer outer = traits::future_access<Outer>::create(outer_state);

                // take by value, as the future will go away immediately
                inner_shared_state_ptr inner_state =
                    traits::future_access<inner_future>::get_shared_state(outer.get());

                if (inner_state.get() == 0)
                {
                    HPX_THROW_EXCEPTION(no_state,
                        "unwrap_continuation<ContResult>::on_outer_ready",
                        "the inner future has no valid shared state");
                }

                inner_state->set_on_completed(
                    util::bind(inner_ready, std::move(this_), inner_state));
            }
            catch (...) {
                this->set_exception(boost::current_exception());
            }
        }

    public:
        template <typename Future>
        void attach(Future& future)
        {
            typedef
                typename shared_state_ptr_for<Future>::type
                outer_shared_state_ptr;

            // Bind an on_completed handler to this future which will wait for
            // the inner future and will transfer its result to the new future.
            boost::intrusive_ptr<unwrap_continuation> this_(this);
            void (unwrap_continuation::*outer_ready)(
                outer_shared_state_ptr const&) =
                    &unwrap_continuation::on_outer_ready<Future>;
            
            if (future.wait_for(boost::posix_time::seconds(0)) == future_status::deferred)
                future.wait();

            outer_shared_state_ptr const& outer_state =
                traits::future_access<Future>::get_shared_state(future);
            outer_state->set_on_completed(
                util::bind(outer_ready, std::move(this_), outer_state));
        }
    };

    template <typename Future>
    inline typename shared_state_ptr<
        typename future_unwrap_result<Future>::result_type>::type
    unwrap(Future&& future, error_code& ec)
    {
        typedef typename future_unwrap_result<Future>::result_type result_type;
        typedef detail::unwrap_continuation<result_type> shared_state;

        // create a continuation
        typename shared_state_ptr<result_type>::type p(new shared_state());
        static_cast<shared_state*>(p.get())->attach(future);
        return std::move(p);
    }
}}}

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace lcos { namespace detail
{
    class void_continuation : public future_data<void>
    {
    private:
        template <typename Future>
        void on_ready(
            typename shared_state_ptr_for<Future>::type const& state)
        {
            try {
                (void)state->get_result();
                this->set_result(util::unused);
            }
            catch (...) {
                this->set_exception(boost::current_exception());
            }
        }

    public:
        template <typename Future>
        void attach(Future& future)
        {
            typedef
                typename shared_state_ptr_for<Future>::type
                shared_state_ptr;

            // Bind an on_completed handler to this future which will wait for
            // the inner future and will transfer its result to the new future.
            boost::intrusive_ptr<void_continuation> this_(this);
            void (void_continuation::*ready)(shared_state_ptr const&) =
                &void_continuation::on_ready<Future>;

            if (future.wait_for(boost::posix_time::seconds(0)) == future_status::deferred)
                future.wait();

            shared_state_ptr const& state =
                lcos::detail::get_shared_state(future);
            state->set_on_completed(util::bind(ready, std::move(this_), state));
        }
    };

    template <typename Future>
    inline typename shared_state_ptr<void>::type
    make_void_continuation(Future& future)
    {
        typedef detail::void_continuation void_shared_state;

        // create a continuation
        typename shared_state_ptr<void>::type p(new void_shared_state());
        static_cast<void_shared_state*>(p.get())->attach(future);
        return std::move(p);
    }
}}}

#endif
