//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_LCOS_LOCAL_CONTINUATION_APR_17_2012_0150PM)
#define HPX_LCOS_LOCAL_CONTINUATION_APR_17_2012_0150PM

#include <hpx/hpx_fwd.hpp>
#include <hpx/traits/promise_remote_result.hpp>
#include <hpx/util/decay.hpp>
#include <hpx/util/move.hpp>
#include <hpx/lcos/detail/future_data.hpp>
#include <hpx/lcos/future.hpp>

#include <boost/shared_ptr.hpp>
#include <boost/make_shared.hpp>
#include <boost/intrusive_ptr.hpp>
#include <boost/type_traits/remove_reference.hpp>
#include <boost/move/move.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace lcos { namespace detail
{
    template <typename Func, typename Future, typename Destination>
    void invoke_continuation(Func& func, Future& future, Destination& dest,
        boost::mpl::false_)
    {
        try {
            dest.set_data(func(boost::move(future)));
        }
        catch (...) {
            dest.set_exception(boost::current_exception());
        }
    }

    template <typename Func, typename Future, typename Destination>
    void invoke_continuation(Func& func, Future& future, Destination& dest,
        boost::mpl::true_)
    {
        try {
            func(boost::move(future));
            dest.set_data(util::unused);
        }
        catch (...) {
            dest.set_exception(boost::current_exception());
        }
    }

    template <typename Func, typename Future, typename Destination>
    void invoke_continuation(Func& func, Future& future, Destination& dest)
    {
        typedef typename boost::is_void<
            typename util::result_of<Func(Future)>::type
        >::type predicate;

        invoke_continuation(func, future, dest, predicate());
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename Future, typename F, typename ContResult>
    class continuation : public future_data<ContResult>
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
        continuation(BOOST_FWD_REF(Func) f)
          : started_(false), id_(threads::invalid_thread_id)
          , f_(boost::forward<Func>(f))
        {}

        void run_impl(typename shared_state_ptr_for<Future>::type const& f)
        {
            Future future = detail::future_access::create<Future>(f);
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

            Future future = detail::future_access::create<Future>(f);
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
                HPX_STD_BIND(async_impl_ptr, boost::move(this_), f),
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
                HPX_STD_BIND(async_impl_ptr, boost::move(this_), f),
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

            shared_state_ptr const& state =
                future_access::get_shared_state(future);
            state->set_on_completed(util::bind(cb, boost::move(this_), state));
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

            shared_state_ptr const& state =
                future_access::get_shared_state(future);
            state->set_on_completed(util::bind(cb, boost::move(this_), state, boost::ref(sched)));
        }

    protected:
        bool started_;
        threads::thread_id_type id_;
        typename util::decay<F>::type f_;
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename ContResult, typename Future, typename F>
    inline typename shared_state_ptr<ContResult>::type
    make_continuation(Future& future, BOOST_SCOPED_ENUM(launch) policy,
        BOOST_FWD_REF(F) f)
    {
        typedef detail::continuation<Future, F, ContResult> shared_state;

        // create a continuation
        typename shared_state_ptr<ContResult>::type p(
            new shared_state(boost::forward<F>(f)));
        static_cast<shared_state*>(p.get())->attach(future, policy);
        return boost::move(p);
    }

    template <typename ContResult, typename Future, typename F>
    inline typename shared_state_ptr<ContResult>::type
    make_continuation(Future& future, threads::executor& sched,
        BOOST_FWD_REF(F) f)
    {
        typedef detail::continuation<Future, F, ContResult> shared_state;

        // create a continuation
        typename shared_state_ptr<ContResult>::type p(
            new shared_state(boost::forward<F>(f)));
        static_cast<shared_state*>(p.get())->attach(future, sched);
        return boost::move(p);
    }
}}}

#if defined(HPX_ENABLE_DEPRECATED_FUTURE)
///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace lcos
{
    ///////////////////////////////////////////////////////////////////////////
    // attach a local continuation to this future instance
    template <typename Result>
    template <typename F>
    inline typename detail::future_then_result<future<Result>, F>::type
    future<Result>::then(BOOST_FWD_REF(F) f)
    {
        return then(launch::all, boost::forward<F>(f));
    }

    template <typename Result>
    template <typename F>
    inline typename detail::future_then_result<future<Result>, F>::type
    future<Result>::then(BOOST_SCOPED_ENUM(launch) policy, BOOST_FWD_REF(F) f)
    {
        typedef typename util::result_of<F(future)>::type result_type;
        typedef lcos::detail::future_data<result_type> future_data_type;

        if (!future_data_) {
            HPX_THROW_EXCEPTION(no_state,
                "future<Result>::get",
                "this future has no valid shared state");
        }

        boost::intrusive_ptr<future_data_type> p =
            detail::make_continuation<result_type>(
                *this, policy, boost::forward<F>(f));
        return lcos::detail::make_future_from_data<result_type>(boost::move(p));
    }

    template <typename Result>
    template <typename F>
    inline typename detail::future_then_result<future<Result>, F>::type
    future<Result>::then(threads::executor& sched, BOOST_FWD_REF(F) f)
    {
        typedef typename util::result_of<F(future)>::type result_type;
        typedef lcos::detail::future_data<result_type> future_data_type;

        if (!future_data_) {
            HPX_THROW_EXCEPTION(no_state,
                "future<Result>::get",
                "this future has no valid shared state");
        }

        boost::intrusive_ptr<future_data_type> p =
            detail::make_continuation<result_type>(
                *this, sched, boost::forward<F>(f));
        return lcos::detail::make_future_from_data<result_type>(boost::move(p));
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename F>
    inline typename detail::future_then_result<future<void>, F>::type
    future<void>::then(BOOST_FWD_REF(F) f)
    {
        return then(launch::all, boost::forward<F>(f));
    }

    template <typename F>
    inline typename detail::future_then_result<future<void>, F>::type
    future<void>::then(BOOST_SCOPED_ENUM(launch) policy, BOOST_FWD_REF(F) f)
    {
        typedef typename util::result_of<F(future)>::type result_type;
        typedef lcos::detail::future_data<result_type> future_data_type;

        if (!future_data_) {
            HPX_THROW_EXCEPTION(no_state,
                "future<void>::get",
                "this future has no valid shared state");
        }

        boost::intrusive_ptr<future_data_type> p =
            detail::make_continuation<result_type>(
                *this, policy, boost::forward<F>(f));
        return lcos::detail::make_future_from_data<result_type>(boost::move(p));
    }

    template <typename F>
    inline typename detail::future_then_result<future<void>, F>::type
    future<void>::then(threads::executor& sched, BOOST_FWD_REF(F) f)
    {
        typedef typename util::result_of<F(future)>::type result_type;
        typedef lcos::detail::future_data<result_type> future_data_type;

        if (!future_data_) {
            HPX_THROW_EXCEPTION(no_state,
                "future<void>::get",
                "this future has no valid shared state");
        }

        boost::intrusive_ptr<future_data_type> p =
            detail::make_continuation<result_type>(
                *this, sched, boost::forward<F>(f));
        return lcos::detail::make_future_from_data<result_type>(boost::move(p));
    }

}}
#endif

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace lcos { namespace detail
{
    template <typename Source, typename Destination>
    void transfer_result(Source& src, Destination& dest, boost::mpl::false_)
    {
        dest.set_data(src.get());
    }

    template <typename Source, typename Destination>
    void transfer_result(Source& src, Destination& dest, boost::mpl::true_)
    {
        src.get();
        dest.set_data(util::unused);
    }

    template <typename Source, typename Destination>
    void transfer_result(Source& src, Destination& dest)
    {
        typedef typename boost::is_void<
            typename lcos::detail::future_traits<Source>::type
        >::type predicate;

        transfer_result(src, dest, predicate());
    }

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
                Inner inner = future_access::create<Inner>(inner_state);

                transfer_result(inner, *this);
            }
            catch (...) {
                this->set_exception(boost::current_exception());
            }
        }

        template <typename Outer>
        void on_outer_ready(
            typename shared_state_ptr_for<Outer>::type const& outer_state)
        {
            typedef typename future_traits<Outer>::type inner_future;
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
                Outer outer = future_access::create<Outer>(outer_state);

                // take by value, as the future will go away immediately
                inner_shared_state_ptr inner_state =
                    future_access::get_shared_state(outer.get());
                inner_state->set_on_completed(
                    util::bind(inner_ready, boost::move(this_), inner_state));
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

            outer_shared_state_ptr const& outer_state =
                future_access::get_shared_state(future);
            outer_state->set_on_completed(
                util::bind(outer_ready, boost::move(this_), outer_state));
        }
    };

    template <typename Future>
    inline typename shared_state_ptr<
        typename unwrap_result<Future>::type>::type
    unwrap(Future& future, error_code& ec)
    {
        typedef typename unwrap_result<Future>::type result_type;
        typedef detail::unwrap_continuation<result_type> shared_state;

        // create a continuation
        typename shared_state_ptr<result_type>::type p(new shared_state());
        static_cast<shared_state*>(p.get())->attach(future);
        return boost::move(p);
    }
}}}

#if defined(HPX_ENABLE_DEPRECATED_FUTURE)
///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace lcos
{
    template <typename Result>
    inline typename detail::future_unwrap_result<future<Result> >::type
    future<Result>::unwrap(error_code& ec)
    {
        BOOST_STATIC_ASSERT_MSG(
            traits::is_future<Result>::value, "invalid use of unwrap");

        typedef typename lcos::detail::future_traits<Result>::type result_type;
        typedef lcos::detail::future_data<result_type> future_data_type;

        if (!valid()) {
            HPX_THROWS_IF(ec, no_state,
                "future<Result>::unwrap",
                "this future has not been initialized");
            return future<result_type>();
        }

        boost::intrusive_ptr<future_data_type> p = detail::unwrap(*this, ec);
        return lcos::detail::make_future_from_data<result_type>(boost::move(p));
    }
}}
#endif

#endif
