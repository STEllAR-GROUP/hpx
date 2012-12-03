//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_LCOS_LOCAL_PROMISE_MAR_01_2012_0121PM)
#define HPX_LCOS_LOCAL_PROMISE_MAR_01_2012_0121PM

#include <hpx/hpx_fwd.hpp>
#include <hpx/util/move.hpp>
#include <hpx/runtime/applier/applier.hpp>
#include <hpx/lcos/detail/future_data.hpp>
#include <hpx/lcos/future.hpp>

#include <boost/atomic.hpp>
#include <boost/intrusive_ptr.hpp>
#include <boost/utility/enable_if.hpp>
#include <boost/utility/result_of.hpp>
#include <boost/type_traits/remove_reference.hpp>
#include <boost/preprocessor/cat.hpp>
#include <boost/preprocessor/repetition/enum.hpp>
#include <boost/preprocessor/repetition/repeat_from_to.hpp>

namespace hpx { namespace lcos { namespace local
{
    namespace detail
    {
        ///////////////////////////////////////////////////////////////////////
        template <typename Result>
        struct future_object
          : lcos::detail::future_data<Result>
        {
            typedef typename lcos::detail::future_data<Result>::result_type
                result_type;

            future_object()
              : lcos::detail::future_data<Result>()
            {}

            // notify of owner going away, this sets an error as the future's
            // result
            void deleting_owner()
            {
                if (!this->is_ready()) {
                    this->set_error(broken_promise,
                        "future_object<Result>::deleting_owner",
                        "deleting owner before future has become ready");
                }
            }
        };

        ///////////////////////////////////////////////////////////////////////
        template <typename Result, typename F>
        struct task_object
          : lcos::detail::task_base<Result>
        {
            typedef typename lcos::detail::task_base<Result>::result_type
                result_type;

            F f_;

            task_object(F const& f)
              : f_(f)
            {}

            task_object(BOOST_RV_REF(F) f)
              : f_(boost::move(f))
            {}

            void do_run()
            {
                try {
                    this->set_data(f_());
                }
                catch(...) {
                    this->set_exception(boost::current_exception());
                }
            }
        };

        template <typename F>
        struct task_object<void, F>
          : lcos::detail::task_base<void>
        {
            typedef typename lcos::detail::task_base<void>::result_type
                result_type;

            F f_;

            task_object(F const& f)
              : f_(f)
            {}

            task_object(BOOST_RV_REF(F) f)
              : f_(boost::move(f))
            {}

            void do_run()
            {
                try {
                    f_();
                    this->set_data(result_type());
                }
                catch(...) {
                    this->set_exception(boost::current_exception());
                }
            }
        };
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename Result>
    class promise
    {
    protected:
        typedef lcos::detail::future_data<Result> task_impl_type;

    private:
        BOOST_MOVABLE_BUT_NOT_COPYABLE(promise)

    public:
        // construction and destruction
        promise()
          : future_obtained_(false)
        {}

        ~promise()
        {
            if (task_)
            {
                if (task_->is_ready() && !future_obtained_)
                {
                    task_->set_error(broken_promise,
                        "promise<Result>::~promise()",
                        "deleting owner before future has been retrieved");
                }
                task_->deleting_owner();
            }
        }

        // Assignment
        promise(BOOST_RV_REF(promise) rhs)
          : task_(boost::move(rhs.task_)),
            future_obtained_(rhs.future_obtained_)
        {
            rhs.task_.reset();
            rhs.future_obtained_ = false;
        }

        promise& operator=(BOOST_RV_REF(promise) rhs)
        {
            if (this != &rhs) {
                if (task_)
                {
                    if (task_->is_ready() && !future_obtained_)
                    {
                        task_->set_error(broken_promise,
                            "promise<Result>::operator=()",
                            "deleting owner before future has been retrieved");
                    }
                    task_->deleting_owner();
                }

                task_ = boost::move(rhs.task_);
                future_obtained_ = rhs.future_obtained_;

                rhs.task_.reset();
                rhs.future_obtained_ = false;
            }
            return *this;
        }

        void swap(promise& other)
        {
            task_.swap(other.task_);
            std::swap(future_obtained_, other.future_obtained_);
        }

        // Result retrieval
        lcos::future<Result> get_future(error_code& ec = throws)
        {
            if (!task_) {
                task_ = new detail::future_object<Result>();
                future_obtained_ = false;
            }

            if (future_obtained_) {
                HPX_THROWS_IF(ec, future_already_retrieved,
                    "promise<Result>::get_future",
                    "future already has been retrieved from this promise");
                return lcos::future<Result>();
            }

            future_obtained_ = true;
            return lcos::future<Result>(task_);
        }

//         template <typename F>
//         void set_wait_callback(F f)
//         {
//             task_->set_wait_callback(f, this);
//         }

        template <typename T>
        void set_value(BOOST_FWD_REF(T) result)
        {
            if (!task_ && future_obtained_) {
                HPX_THROW_EXCEPTION(task_moved,
                    "promise<Result>::set_value<T>",
                    "promise invalid (has it been moved?)");
                return;
            }

            if (!task_)
                task_ = new detail::future_object<Result>();

            task_->set_data(boost::forward<T>(result));
        }

        void set_exception(boost::exception_ptr const& e)
        {
            if (!task_ && future_obtained_) {
                HPX_THROW_EXCEPTION(task_moved,
                    "promise<Result>::set_exception",
                    "promise invalid (has it been moved?)");
                return;
            }

            if (!task_)
                task_ = new detail::future_object<Result>();

            task_->set_exception(e);
        }

        bool valid() const BOOST_NOEXCEPT
        {
            return task_;
        }

        bool is_ready() const
        {
            return task_->is_ready();
        }

    protected:
        boost::intrusive_ptr<task_impl_type> task_;
        bool future_obtained_;
    };

    ///////////////////////////////////////////////////////////////////////////
    template <>
    class promise<void>
    {
    private:
        typedef lcos::detail::future_data<void> task_impl_type;

        BOOST_MOVABLE_BUT_NOT_COPYABLE(promise)

    public:
        // construction and destruction
        promise()
          : future_obtained_(false)
        {}

        ~promise()
        {
            if (task_)
            {
                if (task_->is_ready() && !future_obtained_)
                {
                    task_->set_error(broken_promise,
                        "promise<Result>::operator=()",
                        "deleting owner before future has been retrieved");
                }
                task_->deleting_owner();
            }
        }

        // Assignment
        promise(BOOST_RV_REF(promise) rhs)
          : task_(rhs.task_),
            future_obtained_(rhs.future_obtained_)
        {
            rhs.task_.reset();
            rhs.future_obtained_ = false;
        }

        promise& operator=(BOOST_RV_REF(promise) rhs)
        {
            if (this != &rhs) {
                if (task_)
                {
                    if (task_->is_ready() && !future_obtained_)
                    {
                        task_->set_error(broken_promise,
                            "promise<void>::operator=()",
                            "deleting owner before future has been retrieved");
                    }
                    task_->deleting_owner();
                }

                task_ = rhs.task_;
                future_obtained_ = rhs.future_obtained_;

                rhs.task_.reset();
                rhs.future_obtained_ = false;
            }
            return *this;
        }

        void swap(promise& other)
        {
            task_.swap(other.task_);
            std::swap(future_obtained_, other.future_obtained_);
        }

        // Result retrieval
        lcos::future<void> get_future(error_code& ec = throws)
        {
            if (!task_) {
                task_ = new detail::future_object<void>();
                future_obtained_ = false;
            }

            if (future_obtained_) {
                HPX_THROWS_IF(ec, future_already_retrieved,
                    "promise<void>::get_future",
                    "future already has been retrieved from this promise");
                return lcos::future<void>();
            }

            future_obtained_ = true;
            return lcos::future<void>(task_);
        }

//         template <typename F>
//         void set_wait_callback(F f)
//         {
//             task_->set_wait_callback(f, this);
//         }

        void set_value()
        {
            if (!task_ && future_obtained_) {
                HPX_THROW_EXCEPTION(task_moved,
                    "promise<void>::set_value",
                    "promise invalid (has it been moved?)");
                return;
            }

            if (!task_)
                task_ = new detail::future_object<void>();

            task_->set_data(util::unused);
        }

        void set_exception(boost::exception_ptr const& e)
        {
            if (!task_ && future_obtained_) {
                HPX_THROW_EXCEPTION(task_moved,
                    "promise<void>::set_exception",
                    "promise invalid (has it been moved?)");
                return;
            }

            if (!task_)
                task_ = new detail::future_object<void>();

            task_->set_exception(e);
        }

        bool valid() const BOOST_NOEXCEPT
        {
            return task_;
        }

        bool is_ready() const
        {
            return task_->is_ready();
        }

    private:
        boost::intrusive_ptr<task_impl_type> task_;
        bool future_obtained_;
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename Func>
    class packaged_task;

    template <typename Result>
    class packaged_task<Result()>
    {
    protected:
        typedef lcos::detail::task_base<Result> task_impl_type;

    private:
        BOOST_MOVABLE_BUT_NOT_COPYABLE(packaged_task)

    public:
        // support for result_of
        typedef Result result_type;

        // construction and destruction
        packaged_task() {}

        template <typename F>
        explicit packaged_task(BOOST_FWD_REF(F) f)
          : task_(new detail::task_object<Result, F>(boost::forward<F>(f))),
            future_obtained_(false)
        {}

        explicit packaged_task(Result (*f)())
          : task_(new detail::task_object<Result , Result (*)()>(f)),
            future_obtained_(false)
        {}

        ~packaged_task()
        {
            if (task_)
            {
                if (task_->is_ready() && !future_obtained_)
                {
                    task_->set_error(broken_promise,
                        "packaged_task<Result()>::operator=()",
                        "deleting owner before future has been retrieved");
                }
                task_->deleting_owner();
            }
        }

        packaged_task(BOOST_RV_REF(packaged_task) rhs)
          : task_(boost::move(rhs.task_)),
            future_obtained_(rhs.future_obtained_)
        {
            rhs.task_.reset();
            rhs.future_obtained_ = false;
        }

        packaged_task& operator=(BOOST_RV_REF(packaged_task) rhs)
        {
            if (this != &rhs) {
                if (task_)
                {
                    if (task_->is_ready() && !future_obtained_)
                    {
                        task_->set_error(broken_promise,
                            "packaged_task<Result()>::operator=()",
                            "deleting owner before future has been retrieved");
                    }
                    task_->deleting_owner();
                }

                task_ = boost::move(rhs.task_);
                future_obtained_ = rhs.future_obtained_;

                rhs.task_.reset();
                rhs.future_obtained_ = false;
            }
            return *this;
        }

        // synchronous execution
        void operator()()
        {
            if (!task_) {
                HPX_THROW_EXCEPTION(task_moved,
                    "packaged_task<Result()>::operator()",
                    "packaged_task invalid (has it been moved?)");
                return;
            }
            task_->run();
        }

        void swap(packaged_task& other)
        {
            task_.swap(other.task_);
            std::swap(future_obtained_, other.future_obtained_);
        }

        // Result retrieval
        lcos::future<Result> get_future(error_code& ec = throws)
        {
            if (!task_) {
                HPX_THROWS_IF(ec, task_moved,
                    "packaged_task<Result()>::get_future",
                    "packaged_task invalid (has it been moved?)");
                return lcos::future<Result>();
            }
            if (future_obtained_) {
                HPX_THROWS_IF(ec, future_already_retrieved,
                    "packaged_task<Result()>::get_future",
                    "future already has been retrieved from this promise");
                return lcos::future<Result>();
            }

            future_obtained_ = true;
            return lcos::future<Result>(task_);
        }

        bool valid() const BOOST_NOEXCEPT
        {
            return task_;
        }

    protected:
        boost::intrusive_ptr<task_impl_type> task_;
        bool future_obtained_;
    };

    ///////////////////////////////////////////////////////////////////////////
    // The futures_factory is very similar to a packaged_task except that it
    // allows for the owner to go out of scope before the future becomes ready.
    // We provide this class to avoid semantic differences to the C++11
    // std::packaged_task, while otoh it is a very convenient way for us to
    // implement hpx::async.
    template <typename Func>
    class futures_factory;

    template <typename Result>
    class futures_factory<Result()>
    {
    protected:
        typedef lcos::detail::task_base<Result> task_impl_type;

    private:
        BOOST_MOVABLE_BUT_NOT_COPYABLE(futures_factory)

    public:
        // support for result_of
        typedef Result result_type;

        // construction and destruction
        futures_factory() {}

        template <typename F>
        explicit futures_factory(BOOST_FWD_REF(F) f)
          : task_(new detail::task_object<Result, F>(boost::forward<F>(f))),
            future_obtained_(false)
        {}

        explicit futures_factory(Result (*f)())
          : task_(new detail::task_object<Result , Result (*)()>(f)),
            future_obtained_(false)
        {}

        ~futures_factory()
        {
            if (task_ && !future_obtained_)
                task_->deleting_owner();
        }

        futures_factory(BOOST_RV_REF(futures_factory) rhs)
          : task_(boost::move(rhs.task_)),
            future_obtained_(rhs.future_obtained_)
        {
            rhs.task_.reset();
            rhs.future_obtained_ = false;
        }

        futures_factory& operator=(BOOST_RV_REF(futures_factory) rhs)
        {
            if (this != &rhs) {
                if (task_ && !future_obtained_)
                    task_->deleting_owner();

                task_ = boost::move(rhs.task_);
                future_obtained_ = rhs.future_obtained_;

                rhs.task_.reset();
                rhs.future_obtained_ = false;
            }
            return *this;
        }

        // synchronous execution
        void operator()()
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
        void apply()
        {
            if (!task_) {
                HPX_THROW_EXCEPTION(task_moved,
                    "futures_factory<Result()>::apply()",
                    "futures_factory invalid (has it been moved?)");
                return;
            }
            task_->apply();
        }

        void swap(futures_factory& other)
        {
            task_.swap(other.task_);
            std::swap(future_obtained_, other.future_obtained_);
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
            return lcos::future<Result>(task_);
        }

        bool valid() const BOOST_NOEXCEPT
        {
            return task_;
        }

    protected:
        boost::intrusive_ptr<task_impl_type> task_;
        bool future_obtained_;
    };
}}}

#endif
