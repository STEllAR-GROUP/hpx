//  Copyright (c) 2007-2013 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_LCOS_LOCAL_PROMISE_MAR_01_2012_0121PM)
#define HPX_LCOS_LOCAL_PROMISE_MAR_01_2012_0121PM

#include <hpx/hpx_fwd.hpp>
#include <hpx/lcos/detail/future_data.hpp>
#include <hpx/lcos/future.hpp>
#include <hpx/runtime/threads/thread_executor.hpp>
#include <hpx/util/move.hpp>
#include <hpx/util/result_of.hpp>

#include <boost/atomic.hpp>
#include <boost/intrusive_ptr.hpp>
#include <boost/utility/enable_if.hpp>
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
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename Result>
    class promise
    {
    protected:
        typedef lcos::detail::future_data<Result> task_impl_type;

    private:
        HPX_MOVABLE_BUT_NOT_COPYABLE(promise)
        typedef lcos::local::spinlock mutex_type;

    public:
        // construction and destruction
        promise()
          : future_obtained_(false)
        {}

        ~promise()
        {
            typename mutex_type::scoped_lock l(mtx_);
            if (task_)
            {
                if (task_->is_ready_locked() && !future_obtained_)
                {
                    task_->set_error(broken_promise,
                        "promise<Result>::~promise()",
                        "deleting owner before future has been retrieved");
                }
                task_->deleting_owner();
            }
        }

        // Assignment
        promise(promise && rhs)
          : future_obtained_(false)
        {
            typename mutex_type::scoped_lock l(rhs.mtx_);
            task_ = std::move(rhs.task_);
            future_obtained_ = rhs.future_obtained_;
            rhs.future_obtained_ = false;
            rhs.task_.reset();
        }

        promise& operator=(promise && rhs)
        {
            if (this != &rhs) {
                typename mutex_type::scoped_lock l(rhs.mtx_);

                if (task_)
                {
                    if (task_->is_ready_locked() && !future_obtained_)
                    {
                        task_->set_error(broken_promise,
                            "promise<Result>::operator=()",
                            "deleting owner before future has been retrieved");
                    }
                    task_->deleting_owner();
                }

                task_ = std::move(rhs.task_);
                future_obtained_ = rhs.future_obtained_;
                rhs.future_obtained_ = false;
                rhs.task_.reset();
            }
            return *this;
        }

        // Result retrieval
        lcos::future<Result> get_future(error_code& ec = throws)
        {
            typename mutex_type::scoped_lock l(mtx_);

            if (!task_) {
                future_obtained_ = false;
                task_ = new detail::future_object<Result>();
            }

            if (future_obtained_) {
                HPX_THROWS_IF(ec, future_already_retrieved,
                    "promise<Result>::get_future",
                    "future already has been retrieved from this promise");
                return lcos::future<Result>();
            }

            future_obtained_ = true;

            using traits::future_access;
            return future_access<future<Result> >::create(task_);
        }

        template <typename T>
        void set_value(T && result)
        {
            typename mutex_type::scoped_lock l(mtx_);

            if (!task_ && future_obtained_) {
                HPX_THROW_EXCEPTION(task_moved,
                    "promise<Result>::set_value<T>",
                    "promise invalid (has it been moved?)");
                return;
            }

            if (!task_)
                task_ = new detail::future_object<Result>();

            task_->set_result(std::forward<T>(result));
        }

        void set_exception(boost::exception_ptr const& e)
        {
            typename mutex_type::scoped_lock l(mtx_);

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
            typename mutex_type::scoped_lock l(mtx_);
            // avoid warning about conversion to bool
            return task_.get() ? true : false;
        }

        bool is_ready() const
        {
            typename mutex_type::scoped_lock l(mtx_);
            return task_->is_ready_locked();
        }

#ifndef BOOST_NO_CXX11_EXPLICIT_CONVERSION_OPERATORS
        // [N3722, 4.1] asks for this...
        explicit operator lcos::future<Result>()
        {
            return get_future();
        }

        explicit operator lcos::shared_future<Result>()
        {
            return get_future();
        }
#endif

    protected:
        boost::intrusive_ptr<task_impl_type> task_;
        bool future_obtained_;
        mutable mutex_type mtx_;
    };

    ///////////////////////////////////////////////////////////////////////////
    template <>
    class promise<void>
    {
    protected:
        typedef lcos::detail::future_data<void> task_impl_type;

    private:
        HPX_MOVABLE_BUT_NOT_COPYABLE(promise)
        typedef lcos::local::spinlock mutex_type;

    public:
        // construction and destruction
        promise()
          : future_obtained_(false)
        {}

        ~promise()
        {
            mutex_type::scoped_lock l(mtx_);

            if (task_)
            {
                if (task_->is_ready_locked() && !future_obtained_)
                {
                    task_->set_error(broken_promise,
                        "promise<Result>::operator=()",
                        "deleting owner before future has been retrieved");
                }
                task_->deleting_owner();
            }
        }

        // Assignment
        promise(promise && rhs)
          : future_obtained_(false)
        {
            mutex_type::scoped_lock l(rhs.mtx_);

            task_ = std::move(rhs.task_);
            future_obtained_ = rhs.future_obtained_;
            rhs.future_obtained_ = false;
            rhs.task_.reset();
        }

        promise& operator=(promise && rhs)
        {
            if (this != &rhs) {
                mutex_type::scoped_lock l(rhs.mtx_);

                if (task_)
                {
                    if (task_->is_ready_locked() && !future_obtained_)
                    {
                        task_->set_error(broken_promise,
                            "promise<void>::operator=()",
                            "deleting owner before future has been retrieved");
                    }
                    task_->deleting_owner();
                }

                task_ = rhs.task_;
                future_obtained_ = rhs.future_obtained_;

                rhs.future_obtained_ = false;
                rhs.task_.reset();
            }
            return *this;
        }

        // Result retrieval
        lcos::future<void> get_future(error_code& ec = throws)
        {
            mutex_type::scoped_lock l(mtx_);

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

            using traits::future_access;
            return future_access<future<void> >::create(task_);
        }

        void set_value()
        {
            mutex_type::scoped_lock l(mtx_);

            if (!task_ && future_obtained_) {
                HPX_THROW_EXCEPTION(task_moved,
                    "promise<void>::set_value",
                    "promise invalid (has it been moved?)");
                return;
            }

            if (!task_)
                task_ = new detail::future_object<void>();

            task_->set_result(util::unused);
        }

        void set_exception(boost::exception_ptr const& e)
        {
            mutex_type::scoped_lock l(mtx_);

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
            mutex_type::scoped_lock l(mtx_);
            // avoid warning about conversion to bool
            return task_.get() ? true : false;
        }

        bool is_ready() const
        {
            mutex_type::scoped_lock l(mtx_);
            return task_->is_ready_locked();
        }

#ifndef BOOST_NO_CXX11_EXPLICIT_CONVERSION_OPERATORS
        // [N3722, 4.1] asks for this...
        explicit operator lcos::future<void>()
        {
            return get_future();
        }

        explicit operator lcos::shared_future<void>()
        {
            return get_future();
        }
#endif

    private:
        boost::intrusive_ptr<task_impl_type> task_;
        bool future_obtained_;
        mutable mutex_type mtx_;
    };
}}}

#ifdef BOOST_NO_CXX11_EXPLICIT_CONVERSION_OPERATORS
namespace hpx { namespace lcos
{
    // [N3722, 4.1] asks for this...
    template <typename Result>
    inline future<Result>::future(local::promise<Result>& promise)
    {
        promise.get_future().swap(*this);
    }

    template <typename Result>
    inline shared_future<Result>::shared_future(local::promise<Result>& promise)
    {
        shared_future<Result>(promise.get_future()).swap(*this);
    }

    // [N3722, 4.1] asks for this...
    template <>
    inline future<void>::future(local::promise<void>& promise)
    {
        promise.get_future().swap(*this);
    }

    template <>
    inline shared_future<void>::shared_future(local::promise<void>& promise)
    {
        shared_future<void>(promise.get_future()).swap(*this);
    }
}}
#endif

#endif
