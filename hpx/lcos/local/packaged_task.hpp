//  Copyright (c) 2007-2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_LCOS_LOCAL_PACKAGED_TASK_MAR_01_2012_0121PM)
#define HPX_LCOS_LOCAL_PACKAGED_TASK_MAR_01_2012_0121PM

#include <hpx/config.hpp>
#include <hpx/lcos/future.hpp>
#include <hpx/lcos/local/promise.hpp>
#include <hpx/lcos/detail/future_data.hpp>
#include <hpx/runtime/threads/thread_executor.hpp>
#include <hpx/runtime/launch_policy.hpp>
#include <hpx/traits/is_callable.hpp>
#include <hpx/util/decay.hpp>
#include <hpx/util/deferred_call.hpp>
#include <hpx/util/thread_description.hpp>

#include <boost/intrusive_ptr.hpp>
#include <boost/mpl/bool.hpp>
#include <boost/type_traits/is_same.hpp>
#include <boost/type_traits/is_void.hpp>
#include <boost/utility/enable_if.hpp>

namespace hpx { namespace lcos { namespace local
{
    namespace detail
    {
        ///////////////////////////////////////////////////////////////////////
        template <typename Result, typename F,
            typename Base = lcos::detail::task_base<Result> >
        struct task_object : Base
        {
            typedef Base base_type;
            typedef typename Base::result_type result_type;

            F f_;

            task_object(F const& f)
              : f_(f)
            {}

            task_object(F && f)
              : f_(std::move(f))
            {}

            task_object(threads::executor& sched, F const& f)
              : base_type(sched), f_(f)
            {}

            task_object(threads::executor& sched, F && f)
              : base_type(sched), f_(std::move(f))
            {}

            void do_run()
            {
                try {
                    this->set_value(f_());
                }
                catch(...) {
                    this->set_exception(boost::current_exception());
                }
            }

        protected:
            // run in a separate thread
            void apply(launch policy,
                threads::thread_priority priority,
                threads::thread_stacksize stacksize, error_code& ec)
            {
                this->check_started();

                typedef typename Base::future_base_type future_base_type;
                future_base_type this_(this);

                util::thread_description desc(f_);   // inherit description
                if (this->sched_) {
                    this->sched_->add(
                        util::bind(&base_type::run_impl, std::move(this_)),
                        desc, threads::pending, false, stacksize, ec);
                }
                else if (policy == launch::fork) {
                    threads::register_thread_plain(
                        util::bind(&base_type::run_impl, std::move(this_)),
                        desc, threads::pending, false,
                        threads::thread_priority_boost,
                        get_worker_thread_num(), stacksize, ec);
                }
                else {
                    threads::register_thread_plain(
                        util::bind(&base_type::run_impl, std::move(this_)),
                        desc, threads::pending, false, priority, std::size_t(-1),
                        stacksize, ec);
                }
            }
        };

        template <typename F, typename Base>
        struct task_object<void, F, Base> : Base
        {
            typedef Base base_type;
            typedef typename Base::result_type result_type;

            F f_;

            task_object(F const& f)
              : f_(f)
            {}

            task_object(F && f)
              : f_(std::move(f))
            {}

            task_object(threads::executor& sched, F const& f)
              : base_type(sched), f_(f)
            {}

            task_object(threads::executor& sched, F && f)
              : base_type(sched), f_(std::move(f))
            {}

            void do_run()
            {
                try {
                    f_();
                    this->set_value(result_type());
                }
                catch(...) {
                    this->set_exception(boost::current_exception());
                }
            }

        protected:
            // run in a separate thread
            void apply(launch policy,
                threads::thread_priority priority,
                threads::thread_stacksize stacksize, error_code& ec)
            {
                this->check_started();

                typedef typename Base::future_base_type future_base_type;
                future_base_type this_(this);

                util::thread_description desc(f_);   // inherit description
                if (this->sched_) {
                    this->sched_->add(
                        util::bind(&base_type::run_impl, std::move(this_)),
                        desc, threads::pending, false, stacksize, ec);
                }
                else if (policy == launch::fork) {
                    threads::register_thread_plain(
                        util::bind(&base_type::run_impl, std::move(this_)),
                        desc, threads::pending, false,
                        threads::thread_priority_boost, get_worker_thread_num(),
                        stacksize, ec);
                }
                else {
                    threads::register_thread_plain(
                        util::bind(&base_type::run_impl, std::move(this_)),
                        desc, threads::pending, false, priority, std::size_t(-1),
                        stacksize, ec);
                }
            }
        };

        template <typename Result, typename F>
        struct cancelable_task_object
          : task_object<Result, F, lcos::detail::cancelable_task_base<Result> >
        {
            typedef task_object<
                    Result, F, lcos::detail::cancelable_task_base<Result>
                > base_type;
            typedef typename base_type::result_type result_type;

            cancelable_task_object(F const& f)
              : base_type(f)
            {}

            cancelable_task_object(F && f)
              : base_type(std::move(f))
            {}

            cancelable_task_object(threads::executor& sched, F const& f)
              : base_type(sched, f)
            {}

            cancelable_task_object(threads::executor& sched, F && f)
              : base_type(sched, std::move(f))
            {}
        };
    }

    ///////////////////////////////////////////////////////////////////////////
    // The futures_factory is very similar to a packaged_task except that it
    // allows for the owner to go out of scope before the future becomes ready.
    // We provide this class to avoid semantic differences to the C++11
    // std::packaged_task, while otoh it is a very convenient way for us to
    // implement hpx::async.
    template <typename Func, bool Cancelable = false>
    class futures_factory;

    namespace detail
    {
        template <typename Result, bool Cancelable>
        struct create_task_object
        {
            typedef boost::intrusive_ptr<lcos::detail::task_base<Result> >
                return_type;

            template <typename F>
            static return_type call(threads::executor& sched, F && f)
            {
                return new task_object<Result, F>(sched, std::forward<F>(f));
            }

            template <typename R>
            static return_type call(threads::executor& sched, R (*f)())
            {
                return new task_object<Result, Result (*)()>(sched, f);
            }

            template <typename F>
            static return_type call(F && f)
            {
                return new task_object<Result, F>(std::forward<F>(f));
            }

            template <typename R>
            static return_type call(R (*f)())
            {
                return new task_object<Result, Result (*)()>(f);
            }
        };

        template <typename Result>
        struct create_task_object<Result, true>
        {
            typedef boost::intrusive_ptr<lcos::detail::task_base<Result> >
                return_type;

            template <typename F>
            static return_type call(threads::executor& sched, F && f)
            {
                return new cancelable_task_object<Result, F>(
                    sched, std::forward<F>(f));
            }

            template <typename R>
            static return_type call(threads::executor& sched, R (*f)())
            {
                return new cancelable_task_object<Result, Result (*)()>(sched, f);
            }

            template <typename F>
            static return_type call(F && f)
            {
                return new cancelable_task_object<Result, F>(std::forward<F>(f));
            }

            template <typename R>
            static return_type call(R (*f)())
            {
                return new cancelable_task_object<Result, Result (*)()>(f);
            }
        };
    }

    template <typename Result, bool Cancelable>
    class futures_factory<Result(), Cancelable>
    {
    protected:
        typedef lcos::detail::task_base<Result> task_impl_type;

    private:
        HPX_MOVABLE_BUT_NOT_COPYABLE(futures_factory)

    public:
        // support for result_of
        typedef Result result_type;

        // construction and destruction
        futures_factory() {}

        template <typename F>
        explicit futures_factory(threads::executor& sched, F && f)
          : task_(detail::create_task_object<Result, Cancelable>::call(
                sched, std::forward<F>(f))),
            future_obtained_(false)
        {}

        explicit futures_factory(threads::executor& sched, Result (*f)())
          : task_(detail::create_task_object<Result, Cancelable>::call(sched, f)),
            future_obtained_(false)
        {}

        template <typename F>
        explicit futures_factory(F && f)
          : task_(detail::create_task_object<Result, Cancelable>::call(
                std::forward<F>(f))),
            future_obtained_(false)
        {}

        explicit futures_factory(Result (*f)())
          : task_(detail::create_task_object<Result, Cancelable>::call(f)),
            future_obtained_(false)
        {}

        ~futures_factory()
        {}

        futures_factory(futures_factory && rhs)
          : task_(std::move(rhs.task_)),
            future_obtained_(rhs.future_obtained_)
        {
            rhs.task_.reset();
            rhs.future_obtained_ = false;
        }

        futures_factory& operator=(futures_factory && rhs)
        {
            if (this != &rhs) {
                task_ = std::move(rhs.task_);
                future_obtained_ = rhs.future_obtained_;

                rhs.task_.reset();
                rhs.future_obtained_ = false;
            }
            return *this;
        }

        // synchronous execution
        void operator()() const
        {
            if (!task_) {
                HPX_THROW_EXCEPTION(task_moved,
                    "futures_factory<Result()>::operator()",
                    "futures_factory invalid (has it been moved?)");
                return;
            }
            task_->run();
        }

        // asynchronous execution
        void apply(
            launch policy = launch::async,
            threads::thread_priority priority = threads::thread_priority_default,
            threads::thread_stacksize stacksize = threads::thread_stacksize_default,
            error_code& ec = throws) const
        {
            if (!task_) {
                HPX_THROW_EXCEPTION(task_moved,
                    "futures_factory<Result()>::apply()",
                    "futures_factory invalid (has it been moved?)");
                return;
            }
            task_->apply(policy, priority, stacksize, ec);
        }

        // Result retrieval
        lcos::future<Result> get_future(error_code& ec = throws)
        {
            if (!task_) {
                HPX_THROWS_IF(ec, task_moved,
                    "futures_factory<Result()>::get_future",
                    "futures_factory invalid (has it been moved?)");
                return lcos::future<Result>();
            }
            if (future_obtained_) {
                HPX_THROWS_IF(ec, future_already_retrieved,
                    "futures_factory<Result()>::get_future",
                    "future already has been retrieved from this promise");
                return lcos::future<Result>();
            }

            future_obtained_ = true;

            using traits::future_access;
            return future_access<future<Result> >::create(task_);
        }

        bool valid() const HPX_NOEXCEPT
        {
            return !!task_;
        }

    protected:
        boost::intrusive_ptr<task_impl_type> task_;
        bool future_obtained_;
    };

    namespace detail
    {
        ///////////////////////////////////////////////////////////////////////
        template <typename Result, typename Signature>
        class packaged_task_base
        {
        private:
            HPX_MOVABLE_BUT_NOT_COPYABLE(packaged_task_base)

        public:
            // construction and destruction
            packaged_task_base()
              : function_()
              , promise_()
            {}

            explicit packaged_task_base(util::function_nonser<Signature> const& f)
              : function_(f)
              , promise_()
            {}

            explicit packaged_task_base(util::function_nonser<Signature> && f)
              : function_(std::move(f))
              , promise_()
            {}

            packaged_task_base(packaged_task_base && other)
              : function_(std::move(other.function_))
              , promise_(std::move(other.promise_))
            {}

            packaged_task_base& operator=(packaged_task_base && other)
            {
                if (this != &other)
                {
                    function_ = std::move(other.function_);
                    promise_ = std::move(other.promise_);
                }
                return *this;
            }

            void swap(packaged_task_base& other) HPX_NOEXCEPT
            {
                function_.swap(other.function_);
                promise_.swap(other.promise_);
            }

            // synchronous execution
            template <typename ...Ts>
            void invoke(boost::mpl::false_, Ts&&... vs)
            {
                if (function_.empty())
                {
                    HPX_THROW_EXCEPTION(no_state,
                        "packaged_task_base<Signature>::get_future",
                        "this packaged_task has no valid shared state");
                    return;
                }

                try {
                    promise_.set_value(function_(std::forward<Ts>(vs)...));
                }
                catch(...) {
                    promise_.set_exception(boost::current_exception());
                }
            }

            template <typename ...Ts>
            void invoke(boost::mpl::true_, Ts&&... vs)
            {
                if (function_.empty())
                {
                    HPX_THROW_EXCEPTION(no_state,
                        "packaged_task_base<Signature>::get_future",
                        "this packaged_task has no valid shared state");
                    return;
                }

                try {
                    function_(std::forward<Ts>(vs)...);
                    promise_.set_value();
                }
                catch(...) {
                    promise_.set_exception(boost::current_exception());
                }
            }

            // Result retrieval
            lcos::future<Result> get_future(error_code& ec = throws)
            {
                if (function_.empty()) {
                    HPX_THROWS_IF(ec, no_state,
                        "packaged_task_base<Signature>::get_future",
                        "this packaged_task has no valid shared state");
                    return lcos::future<Result>();
                }
                return promise_.get_future();
            }

            bool valid() const HPX_NOEXCEPT
            {
                return !function_.empty() && promise_.valid();
            }

            void reset(error_code& ec = throws)
            {
                if (function_.empty()) {
                    HPX_THROWS_IF(ec, no_state,
                        "packaged_task_base<Signature>::get_future",
                        "this packaged_task has no valid shared state");
                    return;
                }
                promise_ = local::promise<Result>();
            }

        protected:
            util::function_nonser<Signature> function_;

        private:
            local::promise<Result> promise_;
        };
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename Sig>
    class packaged_task;

    template <typename R, typename ...Ts>
    class packaged_task<R(Ts...)>
      : private detail::packaged_task_base<R, R(Ts...)>
    {
        HPX_MOVABLE_BUT_NOT_COPYABLE(packaged_task)

        typedef detail::packaged_task_base<R, R(Ts...)> base_type;

    public:
        // support for result_of
        typedef R result_type;

        // construction and destruction
        packaged_task()
          : base_type()
        {}

        template <typename F>
        explicit packaged_task(F && f,
            typename boost::enable_if_c<
                !boost::is_same<typename util::decay<F>::type, packaged_task>::value
             && traits::is_callable<typename util::decay<F>::type(Ts...), R>::value
            >::type* = 0)
          : base_type(std::forward<F>(f))
        {}

        packaged_task(packaged_task && other)
          : base_type(std::move(other))
        {}

        packaged_task& operator=(packaged_task && rhs)
        {
            base_type::operator=(std::move(rhs));
            return *this;
        }

        void swap(packaged_task& other) HPX_NOEXCEPT
        {
            base_type::swap(other);
        }

        template <typename ... Vs>
        void operator()(Vs&&... vs)
        {
            base_type::invoke(boost::is_void<R>(), std::forward<Vs>(vs)...);
        }

        // result retrieval
        using base_type::get_future;
        using base_type::valid;
        using base_type::reset;
    };
}}}

#endif
