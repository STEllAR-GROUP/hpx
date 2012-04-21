//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_LCOS_LOCAL_PROMISE_MAR_01_2012_0121PM)
#define HPX_LCOS_LOCAL_PROMISE_MAR_01_2012_0121PM

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/applier/applier.hpp>
#include <hpx/lcos/detail/future_data.hpp>
#include <hpx/lcos/future.hpp>

#include <boost/atomic.hpp>
#include <boost/move/move.hpp>
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
          : task_(new detail::future_object<Result>()),
            future_obtained_(false)
        {}

        ~promise()
        {
            if (task_)
                task_->deleting_owner();
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
            if (!task_) {
                HPX_THROW_EXCEPTION(task_moved,
                    "promise<Result>::set_value<T>",
                    "promise invalid (has it been moved?)");
                return;
            }
            task_->set_data(boost::forward<T>(result));
        }

        void set_exception(boost::exception_ptr const& e)
        {
            if (!task_) {
                HPX_THROW_EXCEPTION(task_moved,
                    "promise<Result>::set_exception",
                    "promise invalid (has it been moved?)");
                return;
            }
            task_->set_exception(e);
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
          : task_(new detail::future_object<void>()),
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
                task_ = new detail::future_object<void>();
                future_obtained_ = false;
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

//         template <typename F>
//         void set_wait_callback(F f)
//         {
//             task_->set_wait_callback(f, this);
//         }

        void set_value()
        {
            if (!task_) {
                HPX_THROW_EXCEPTION(task_moved,
                    "promise<void>::set_value",
                    "promise invalid (has it been moved?)");
                return;
            }
            task_->set_data(util::unused);
        }

        void set_exception(boost::exception_ptr const& e)
        {
            if (!task_) {
                HPX_THROW_EXCEPTION(task_moved,
                    "promise<void>::set_exception",
                    "promise invalid (has it been moved?)");
                return;
            }
            task_->set_exception(e);
        }

        bool valid() const BOOST_NOEXCEPT
        {
            return task_;
        }

    private:
        boost::intrusive_ptr<task_impl_type> task_;
        bool future_obtained_;
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename Result>
    class packaged_task
    {
    protected:
        typedef lcos::detail::task_base<Result> task_impl_type;

    private:
        BOOST_MOVABLE_BUT_NOT_COPYABLE(packaged_task)

    public:
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
                task_->deleting_owner();
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
                    "packaged_task::operator()",
                    "packaged_task invalid (has it been moved?)");
                return;
            }
            task_->run();
        }

#if 0
// #define HPX_LCOS_LOCAL_FWD_REF_PARAMS(Z, N, D)                                \
//     BOOST_FWD_REF(BOOST_PP_CAT(A, N)) BOOST_PP_CAT(a, N)                      \
// /**/
// #define HPX_LCOS_LOCAL_FWD_PARAMS(Z, N, D)                                    \
//     boost::forward<BOOST_PP_CAT(D, N)>(BOOST_PP_CAT(a, N))                    \
// /**/
//
// #define HPX_LCOS_LOCAL_PROMISE_OPERATOR(Z, N, D)                              \
//     template <BOOST_PP_ENUM_PARAMS(N, typename A)>                            \
//     void operator()(BOOST_PP_ENUM(N, HPX_LCOS_LOCAL_FWD_REF_PARAMS, _))       \
//     {                                                                         \
//         if (!task_) {                                                         \
//             HPX_THROWS_IF(ec, task_moved, "packaged_task::operator()",        \
//                 "packaged_task invalid (has it been moved?)");                \
//             return;                                                           \
//         }                                                                     \
//         task_->run(BOOST_PP_ENUM(N, HPX_LCOS_LOCAL_FWD_PARAMS, _));           \
//     }                                                                         \
//     /**/
//
//         BOOST_PP_REPEAT_FROM_TO(
//             1
//           , HPX_FUNCTION_LIMIT
//           , HPX_LCOS_LOCAL_PROMISE_OPERATOR, _
//         )
//
// #undef HPX_LCOS_LOCAL_PROMISE_OPERATOR
// #undef HPX_LCOS_LOCAL_FWD_PARAMS
// #undef HPX_LCOS_LOCAL_FWD_REF_PARAMS
#endif

        // synchronous execution
        void async()
        {
            if (!task_) {
                HPX_THROW_EXCEPTION(task_moved,
                    "packaged_task::async()",
                    "packaged_task invalid (has it been moved?)");
                return;
            }
            task_->async();
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
                    "packaged_task<Result>::get_future",
                    "packaged_task invalid (has it been moved?)");
                return lcos::future<Result>();
            }
            if (future_obtained_) {
                HPX_THROWS_IF(ec, future_already_retrieved,
                    "packaged_task<Result>::get_future",
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
