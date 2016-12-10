//  Copyright (c) 2007-2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_LCOS_LOCAL_FUTURES_FACTORY_HPP
#define HPX_LCOS_LOCAL_FUTURES_FACTORY_HPP

#include <hpx/config.hpp>
#include <hpx/lcos/detail/future_data.hpp>
#include <hpx/lcos/future.hpp>
#include <hpx/runtime/launch_policy.hpp>
#include <hpx/runtime/threads/thread_data_fwd.hpp>
#include <hpx/runtime/threads/thread_enums.hpp>
#include <hpx/runtime/threads/thread_helpers.hpp>
#include <hpx/throw_exception.hpp>
#include <hpx/traits/future_access.hpp>
#include <hpx/util/deferred_call.hpp>
#include <hpx/util/thread_description.hpp>

#include <boost/exception_ptr.hpp>
#include <boost/intrusive_ptr.hpp>

#include <cstddef>
#include <type_traits>
#include <utility>

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
            typedef typename Base::init_no_addref init_no_addref;

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

            task_object(F const& f, init_no_addref no_addref)
              : base_type(no_addref), f_(f)
            {}

            task_object(F && f, init_no_addref no_addref)
              : base_type(no_addref), f_(std::move(f))
            {}

            task_object(threads::executor& sched, F const& f,
                    init_no_addref no_addref)
              : base_type(sched, no_addref), f_(f)
            {}

            task_object(threads::executor& sched, F && f,
                    init_no_addref no_addref)
              : base_type(sched, no_addref), f_(std::move(f))
            {}

            void do_run()
            {
                return do_run_impl(std::is_void<Result>());
            }

        private:
            void do_run_impl(/*is_void=*/std::false_type)
            {
                try {
                    this->set_value(f_());
                }
                catch(...) {
                    this->set_exception(boost::current_exception());
                }
            }

            void do_run_impl(/*is_void=*/std::true_type)
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
            threads::thread_id_type apply(launch policy,
                threads::thread_priority priority,
                threads::thread_stacksize stacksize, error_code& ec)
            {
                this->check_started();

                typedef typename Base::future_base_type future_base_type;
                future_base_type this_(this);

                if (this->sched_) {
                    this->sched_->add(
                        util::deferred_call(&base_type::run_impl, std::move(this_)),
                        util::thread_description(f_),
                        threads::pending, false, stacksize, ec);
                    return threads::invalid_thread_id;
                }
                else if (policy == launch::fork) {
                    return threads::register_thread_nullary(
                        util::deferred_call(&base_type::run_impl, std::move(this_)),
                        util::thread_description(f_),
                        threads::pending_do_not_schedule, true,
                        threads::thread_priority_boost,
                        get_worker_thread_num(), stacksize, ec);
                }
                else {
                    threads::register_thread_nullary(
                        util::deferred_call(&base_type::run_impl, std::move(this_)),
                        util::thread_description(f_),
                        threads::pending, false, priority, std::size_t(-1),
                        stacksize, ec);
                    return threads::invalid_thread_id;
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

            cancelable_task_object(F const& f, init_no_addref no_addref)
              : base_type(f, no_addref)
            {}

            cancelable_task_object(F && f, init_no_addref no_addref)
              : base_type(std::move(f), no_addref)
            {}

            cancelable_task_object(threads::executor& sched, F const& f,
                    init_no_addref no_addref)
              : base_type(sched, f, no_addref)
            {}

            cancelable_task_object(threads::executor& sched, F && f,
                    init_no_addref no_addref)
              : base_type(sched, std::move(f), no_addref)
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
            typedef
                boost::intrusive_ptr<lcos::detail::task_base<Result> >
                return_type;
            typedef
                typename lcos::detail::future_data_refcnt_base::init_no_addref
                init_no_addref;

            template <typename F>
            static return_type call(threads::executor& sched, F && f)
            {
                return return_type(
                    new task_object<Result, F>(
                        sched, std::forward<F>(f), init_no_addref()),
                    false);
            }

            template <typename R>
            static return_type call(threads::executor& sched, R (*f)())
            {
                return return_type(
                    new task_object<Result, Result (*)()>(
                        sched, f, init_no_addref()),
                    false);
            }

            template <typename F>
            static return_type call(F&& f)
            {
                return return_type(
                    new task_object<Result, F>(
                        std::forward<F>(f), init_no_addref()),
                    false);
            }

            template <typename R>
            static return_type call(R (*f)())
            {
                return return_type(
                    new task_object<Result, Result (*)()>(f, init_no_addref()),
                    false);
            }
        };

        template <typename Result>
        struct create_task_object<Result, true>
        {
            typedef
                boost::intrusive_ptr<lcos::detail::task_base<Result> >
                return_type;
            typedef
                typename lcos::detail::future_data_refcnt_base::init_no_addref
                init_no_addref;

            template <typename F>
            static return_type call(threads::executor& sched, F&& f)
            {
                return return_type(
                    new cancelable_task_object<Result, F>(
                        sched, std::forward<F>(f), init_no_addref()),
                    false);
            }

            template <typename R>
            static return_type call(threads::executor& sched, R (*f)())
            {
                return return_type(
                    new cancelable_task_object<Result, Result (*)()>(
                        sched, f, init_no_addref()),
                    false);
            }

            template <typename F>
            static return_type call(F&& f)
            {
                return return_type(
                    new cancelable_task_object<Result, F>(
                        std::forward<F>(f), init_no_addref()),
                    false);
            }

            template <typename R>
            static return_type call(R (*f)())
            {
                return return_type(
                    new cancelable_task_object<Result, Result (*)()>(
                        f, init_no_addref()),
                    false);
            }
        };
    }

    template <typename Result, bool Cancelable>
    class futures_factory<Result(), Cancelable>
    {
    protected:
        typedef lcos::detail::task_base<Result> task_impl_type;

    private:
        HPX_MOVABLE_ONLY(futures_factory);

    public:
        // construction and destruction
        futures_factory()
          : future_obtained_(false)
        {}

        template <typename F>
        explicit futures_factory(threads::executor& sched, F&& f)
          : task_(detail::create_task_object<Result, Cancelable>::call(
                sched, std::forward<F>(f))),
            future_obtained_(false)
        {}

        explicit futures_factory(threads::executor& sched, Result (*f)())
          : task_(detail::create_task_object<Result, Cancelable>::call(sched, f)),
            future_obtained_(false)
        {}

        template <typename F>
        explicit futures_factory(F&& f)
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

        futures_factory(futures_factory&& rhs)
          : task_(std::move(rhs.task_)),
            future_obtained_(rhs.future_obtained_)
        {
            rhs.task_.reset();
            rhs.future_obtained_ = false;
        }

        futures_factory& operator=(futures_factory&& rhs)
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
        threads::thread_id_type apply(
            launch policy = launch::async,
            threads::thread_priority priority = threads::thread_priority_default,
            threads::thread_stacksize stacksize = threads::thread_stacksize_default,
            error_code& ec = throws) const
        {
            if (!task_) {
                HPX_THROW_EXCEPTION(task_moved,
                    "futures_factory<Result()>::apply()",
                    "futures_factory invalid (has it been moved?)");
                return threads::invalid_thread_id;
            }
            return task_->apply(policy, priority, stacksize, ec);
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

        // This is the same as get_future above, except that it moves the
        // shared state into the returned future.
        lcos::future<Result> retrieve_future(error_code& ec = throws)
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
            return future_access<future<Result> >::create(std::move(task_));
        }

        bool valid() const HPX_NOEXCEPT
        {
            return !!task_;
        }

    protected:
        boost::intrusive_ptr<task_impl_type> task_;
        bool future_obtained_;
    };
}}}

#endif /*HPX_LCOS_LOCAL_FUTURES_FACTORY_HPP*/
