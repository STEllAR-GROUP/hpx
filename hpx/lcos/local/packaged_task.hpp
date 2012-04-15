//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_LCOS_LOCAL_promise_MAR_01_2012_0121PM)
#define HPX_LCOS_LOCAL_promise_MAR_01_2012_0121PM

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/applier/applier.hpp>
#include <hpx/lcos/detail/future_data.hpp>

#include <boost/move/move.hpp>
#include <boost/intrusive_ptr.hpp>

namespace hpx { namespace lcos { namespace local
{
    namespace detail
    {
        template<typename Result, typename F>
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

        template<typename F>
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
    private:
        typedef lcos::detail::task_base<Result> task_impl_type;

        BOOST_MOVABLE_BUT_NOT_COPYABLE(promise)

    public:
        // construction and destruction
        promise() : future_obtained_(false) {}

        template <typename F>
        explicit promise(BOOST_FWD_REF(F) f)
          : task_(new detail::task_object<Result, F>(boost::forward<F>(f))),
            future_obtained_(false)
        {}

        explicit promise(Result (*f)())
          : task_(new detail::task_object<Result , Result (*)()>(f)),
            future_obtained_(false)
        {}

        ~promise()
        {
            if (task_)
                task_->deleting_owner();
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
            task_ = rhs.task_;
            future_obtained_ = rhs.future_obtained_;
            rhs.task_.reset();
            rhs.future_obtained_ = false;
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
                HPX_THROWS_IF(ec, task_moved,
                    "promise<Result>::get_future",
                    "task invalid (has it been moved?)");
                return lcos::future<Result>();
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

        // execution
        void operator()(error_code& ec = throws)
        {
            if (!task_) {
                HPX_THROWS_IF(ec, task_moved,
                    "promise::operator()",
                    "task invalid (has it been moved?)");
                return;
            }
            task_->run(ec);
        }

//         template <typename F>
//         void set_wait_callback(F f)
//         {
//             task_->set_wait_callback(f, this);
//         }

        template <typename T>
        void set_value(BOOST_FWD_REF(T) result)
        {
            task_->set_data(boost::forward<T>(result));
        }

        void set_exception(boost::exception_ptr const& e)
        {
            task_->set_exception(e);
        }

    private:
        boost::intrusive_ptr<task_impl_type> task_;
        bool future_obtained_;
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename Result>
    struct packaged_task : promise<Result>
    {
        // construction and destruction
        packaged_task() {}

        template <typename F>
        explicit packaged_task(BOOST_FWD_REF(F) f)
          : promise<Result>(boost::forward<F>(f))
        {
            (*this)();    // execute the function immediately
        }

        explicit packaged_task(Result(*f)())
          : promise<Result>(f)
        {
            (*this)();    // execute the function immediately
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    template <>
    class promise<void>
    {
    private:
        typedef lcos::detail::task_base<void> task_impl_type;

        BOOST_MOVABLE_BUT_NOT_COPYABLE(promise)

    public:
        // construction and destruction
        promise() : future_obtained_(false) {}

        template <typename F>
        explicit promise(BOOST_FWD_REF(F) f)
          : task_(new detail::task_object<void, F>(boost::forward<F>(f))),
            future_obtained_(false)
        {}

        explicit promise(void (*f)())
          : task_(new detail::task_object<void , void (*)()>(f)),
            future_obtained_(false)
        {}

        ~promise()
        {
            if (task_)
                task_->deleting_owner();
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
            task_ = rhs.task_;
            future_obtained_ = rhs.future_obtained_;
            rhs.task_.reset();
            rhs.future_obtained_ = false;
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
                HPX_THROWS_IF(ec, task_moved,
                    "promise<Result>::get_future",
                    "task invalid (has it been moved?)");
                return lcos::future<void>();
            }
            if (future_obtained_) {
                HPX_THROWS_IF(ec, future_already_retrieved,
                    "promise<Result>::get_future",
                    "future already has been retrieved from this promise");
                return lcos::future<void>();
            }

            future_obtained_ = true;
            return lcos::future<void>(task_);
        }

        // execution
        void operator()(error_code& ec = throws)
        {
            if (!task_) {
                HPX_THROWS_IF(ec, task_moved,
                    "promise::operator()",
                    "task invalid (has it been moved?)");
                return;
            }
            task_->run(ec);
        }

//         template <typename F>
//         void set_wait_callback(F f)
//         {
//             task_->set_wait_callback(f, this);
//         }

        void set_value()
        {
            task_->set_data(util::unused);
        }

        void set_exception(boost::exception_ptr const& e)
        {
            task_->set_exception(e);
        }

    private:
        boost::intrusive_ptr<task_impl_type> task_;
        bool future_obtained_;
    };
}}}

#endif
