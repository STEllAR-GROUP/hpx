//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_LCOS_LOCAL_EAGER_FUTURE_MAR_01_2012_0121PM)
#define HPX_LCOS_LOCAL_EAGER_FUTURE_MAR_01_2012_0121PM

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
        struct task_object : lcos::detail::task_base<Result>
        {
            typedef Result result_type;

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
                    this->set_error(boost::current_exception());
                }
            }
        };

        template<typename F>
        struct task_object<void, F> : lcos::detail::task_base<util::unused_type>
        {
            typedef util::unused_type result_type;

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
                    this->set_error(boost::current_exception());
                }
            }
        };
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename Result>
    class packaged_task
    {
    private:
        typedef lcos::detail::task_base<Result> task_impl_type;

        BOOST_MOVABLE_BUT_NOT_COPYABLE(packaged_task)

    public:
        // construction and destruction
        template <typename F>
        explicit packaged_task(F const& f)
          : task_(new detail::task_object<Result, F>(f)),
            future_obtained_(false)
        {}

        template <typename F>
        explicit packaged_task(BOOST_FWD_REF(F) f)
          : task_(new detail::task_object<Result, F>(boost::forward<F>(f))),
            future_obtained_(false)
        {}

        explicit packaged_task(Result(*f)())
          : task_(new detail::task_object<Result , Result (*)()>(f)),
            future_obtained_(false)
        {}

        ~packaged_task()
        {
            if (task_)
                task_->deleting_owner();
        }

        // Assignment
        packaged_task(BOOST_RV_REF(packaged_task) rhs)
          : task_(rhs.future_data_),
            future_obtained_(rhs.future_obtained_)
        {
            rhs.task_.reset();
            rhs.future_obtained_ = false;
        }

        packaged_task& operator=(BOOST_RV_REF(packaged_task) rhs)
        {
            task_ = rhs.task_;
            future_obtained_ = rhs.future_obtained_;
            rhs.task_.reset();
            rhs.future_obtained_ = false;
            return *this;
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
                    "task invalid (has it been moved?)");
                return lcos::future<Result>();
            }
            if (future_obtained_) {
                HPX_THROWS_IF(ec, future_already_retrieved,
                    "packaged_task<Result>::get_future",
                    "future already has been retrieved from this packaged_task");
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
                    "packaged_task::operator()",
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

    private:
        boost::intrusive_ptr<task_impl_type> task_;
        bool future_obtained_;
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename Result>
    struct eager_future : packaged_task<Result>
    {
        // construction and destruction
        template <typename F>
        explicit eager_future(F const& f)
          : packaged_task<Result>(f)
        {
            (*this)();    // execute the function immediately
        }

        template <typename F>
        explicit eager_future(BOOST_FWD_REF(F) f)
          : packaged_task<Result>(boost::forward<F>(f))
        {
            (*this)();    // execute the function immediately
        }

        explicit eager_future(Result(*f)())
          : packaged_task<Result>(f)
        {
            (*this)();    // execute the function immediately
        }
    };
}}}

#endif
