//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_LCOS_LOCAL_CONTINUATION_APR_17_2012_0150PM)
#define HPX_LCOS_LOCAL_CONTINUATION_APR_17_2012_0150PM

#include <hpx/hpx_fwd.hpp>
#include <hpx/lcos/detail/future_data.hpp>
#include <hpx/lcos/future.hpp>

#include <boost/shared_ptr.hpp>
#include <boost/make_shared.hpp>
#include <boost/move/move.hpp>
#include <boost/intrusive_ptr.hpp>
#include <boost/type_traits/remove_reference.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace lcos { namespace detail
{
    template <typename ContResult>
    struct continuation_base : future_data<ContResult>
    {
    private:
        typedef typename future_data<ContResult>::mutex_type mutex_type;
        typedef boost::intrusive_ptr<continuation_base> future_base_type;

    protected:
        typedef typename future_data<ContResult>::result_type result_type;

    protected:
        threads::thread_id_type get_id() const
        {
            typename mutex_type::scoped_lock l(this->mtx_);
            return id_;
        }
        void set_id(threads::thread_id_type id)
        {
            typename mutex_type::scoped_lock l(this->mtx_);
            id_ = id;
        }

        struct reset_id
        {
            reset_id(continuation_base& target)
              : target_(target)
            {
                target.set_id(threads::get_self().get_thread_id());
            }
            ~reset_id()
            {
                target_.set_id(threads::invalid_thread_id);
            }
            continuation_base& target_;
        };

    public:
        continuation_base()
          : started_(false), id_(threads::invalid_thread_id)
        {}

        // retrieving the value
        result_type get(error_code& ec = throws)
        {
            if (!started_) {
                lcos::future<ContResult> f(this);
                run(f, ec);
            }
            return boost::move(this->get_data(ec));
        }

        // moving out the value
        result_type move(error_code& ec = throws)
        {
            if (!started_) {
                lcos::future<ContResult> f(this);
                run(f, ec);
            }
            return boost::move(this->move_data(ec));
        }

        template <typename Result>
        void run_impl(lcos::future<Result> const& f);

        template <typename Result>
        void run (lcos::future<Result> const& f, error_code& ec)
        {
            {
                typename mutex_type::scoped_lock l(this->mtx_);
                if (started_) {
                    HPX_THROWS_IF(ec, task_already_started,
                        "continuation_base::run",
                        "this task has already been started");
                    return;
                }
                started_ = true;
            }

            run_impl<Result>(f);

            if (&ec != &throws)
                ec = make_success_code();
        }

        template <typename Result>
        threads::thread_state_enum
        async_impl(lcos::future<Result> const& f);

        template <typename Result>
        void async (lcos::future<Result> const& f, error_code& ec)
        {
            {
                typename mutex_type::scoped_lock l(this->mtx_);
                if (started_) {
                    HPX_THROWS_IF(ec, task_already_started,
                        "continuation_base::run",
                        "this task has already been started");
                    return;
                }
                started_ = true;
            }

            future_base_type this_(this);
            applier::register_thread_plain(
                HPX_STD_BIND(&continuation_base::async_impl<Result>, this_, f),
                "continuation_base::async");

            if (&ec != &throws)
                ec = make_success_code();
        }

        void deleting_owner()
        {
            typename mutex_type::scoped_lock l(this->mtx_);
            if (!started_) {
                started_ = true;
                l.unlock();
                this->set_error(broken_task,
                    "continuation_base::deleting_owner",
                    "deleting task owner before future has been executed");
            }
        }

        // cancellation support
        bool is_cancelable() const
        {
            return true;
        }

        void cancel()
        {
            typename mutex_type::scoped_lock l(this->mtx_);
            try {
                if (!this->started_) {
                    HPX_THROW_EXCEPTION(thread_interrupted,
                        "continuation_base<Result>::cancel",
                        "future has been canceled");
                    return;
                }

                if (this->is_ready())
                    return;   // nothing we can do

                if (id_ != threads::invalid_thread_id) {
                    // interrupt the executing thread
                    threads::interrupt_thread(id_);

                    this->started_ = true;

                    l.unlock();
                    this->set_error(thread_interrupted,
                        "continuation_base<Result>::cancel",
                        "future has been canceled");
                }
                else {
                    HPX_THROW_EXCEPTION(future_can_not_be_cancelled,
                        "continuation_base<Result>::cancel",
                        "future can't be canceled at this time");
                }
            }
            catch (hpx::exception const&) {
                this->started_ = true;
                this->set_exception(boost::current_exception());
                throw;
            }
        }

    protected:
        bool started_;
        threads::thread_id_type id_;
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename ContResult, typename Result>
    struct continuation : continuation_base<ContResult>
    {
        typedef typename lcos::detail::continuation_base<
            ContResult
        >::result_type result_type;

        virtual void do_run(lcos::future<Result> const& f) = 0;
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename ContResult>
    template <typename Result>
    void continuation_base<ContResult>::run_impl(lcos::future<Result> const& f)
    {
        typedef continuation<ContResult, Result> derived_type;
        static_cast<derived_type*>(this)->do_run(f);
    }

    template <typename ContResult>
    template <typename Result>
    threads::thread_state_enum continuation_base<ContResult>::async_impl(
        lcos::future<Result> const& f)
    {
        typedef continuation<ContResult, Result> derived_type;
        reset_id r(*this);
        static_cast<derived_type*>(this)->do_run(f);
        return threads::terminated;
    }
}}}

namespace hpx { namespace lcos { namespace local
{
    namespace detail
    {
        template <typename ContResult, typename Result, typename F>
        struct continuation_object
          : lcos::detail::continuation<ContResult, Result>
        {
            typedef typename lcos::detail::continuation<
                ContResult, Result
            >::result_type result_type;

            F f_;

            continuation_object(F const& f)
              : f_(f)
            {}

            template <typename Func>
            continuation_object(BOOST_FWD_REF(Func) f)
              : f_(boost::forward<Func>(f))
            {}

            void do_run(lcos::future<Result> const& f)
            {
                try {
                    this->set_data(f_(f));
                }
                catch(...) {
                    this->set_exception(boost::current_exception());
                }
            }
        };

        ///////////////////////////////////////////////////////////////////////
        template <typename Result, typename F>
        struct continuation_object<void, Result, F>
          : lcos::detail::continuation<void, Result>
        {
            typedef
                typename lcos::detail::continuation<void, Result>::result_type
            result_type;

            F f_;

            continuation_object(F const& f)
              : f_(f)
            {}

            template <typename Func>
            continuation_object(BOOST_FWD_REF(Func) f)
              : f_(boost::forward<Func>(f))
            {}

            void do_run(lcos::future<Result> const& f)
            {
                try {
                    f_(f);
                    this->set_data(result_type());
                }
                catch(...) {
                    this->set_exception(boost::current_exception());
                }
            }
        };
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename ContResult, typename Result>
    class packaged_continuation
    {
    protected:
        typedef lcos::detail::continuation_base<ContResult> cont_impl_type;

    private:
        BOOST_MOVABLE_BUT_NOT_COPYABLE(packaged_continuation)

    public:
        // construction and destruction
        packaged_continuation() : future_obtained_(false) {}

        template <typename F>
        explicit packaged_continuation(BOOST_FWD_REF(F) f)
          : cont_(new detail::continuation_object<ContResult, Result, F>(
                boost::forward<F>(f))),
            future_obtained_(false)
        {}

        ~packaged_continuation()
        {
            if (cont_)
                cont_->deleting_owner();
        }

        // Assignment
        packaged_continuation(BOOST_RV_REF(packaged_continuation) rhs)
          : cont_(rhs.cont_),
            future_obtained_(rhs.future_obtained_)
        {
            rhs.cont_.reset();
            rhs.future_obtained_ = false;
        }

        packaged_continuation& operator=(BOOST_RV_REF(packaged_continuation) rhs)
        {
            cont_ = rhs.cont_;
            future_obtained_ = rhs.future_obtained_;
            rhs.cont_.reset();
            rhs.future_obtained_ = false;
            return *this;
        }

        void swap(packaged_continuation& other)
        {
            cont_.swap(other.cont_);
            std::swap(future_obtained_, other.future_obtained_);
        }

        // Result retrieval
        lcos::future<ContResult> get_future(error_code& ec = throws)
        {
            if (!cont_) {
                HPX_THROWS_IF(ec, task_moved,
                    "packaged_continuation<ContResult>::get_future",
                    "task invalid (has it been moved?)");
                return lcos::future<ContResult>();
            }
            if (future_obtained_) {
                HPX_THROWS_IF(ec, future_already_retrieved,
                    "packaged_continuation<ContResult>::get_future",
                    "future already has been retrieved from this promise");
                return lcos::future<ContResult>();
            }

            future_obtained_ = true;
            return lcos::future<ContResult>(cont_);
        }

        // synchronous execution
        void operator()(lcos::future<Result> f, error_code& ec = throws)
        {
            if (!cont_) {
                HPX_THROWS_IF(ec, task_moved,
                    "packaged_continuation::operator()",
                    "task invalid (has it been moved?)");
                return;
            }
            cont_->run(f, ec);
        }

        // asynchronous execution
        void async(lcos::future<Result> f, error_code& ec = throws)
        {
            if (!cont_) {
                HPX_THROWS_IF(ec, task_moved,
                    "packaged_continuation::async()",
                    "task invalid (has it been moved?)");
                return;
            }
            cont_->async(f, ec);
        }

//         template <typename F>
//         void set_wait_callback(F f)
//         {
//             task_->set_wait_callback(f, this);
//         }

        template <typename T>
        void set_value(BOOST_FWD_REF(T) result)
        {
            cont_->set_data(boost::forward<T>(result));
        }

        void set_exception(boost::exception_ptr const& e)
        {
            cont_->set_exception(e);
        }

        void on_value_ready(lcos::future<Result> const& f)
        {
            (*this)(f);   // pass this future on to the continuation
        }

    private:
        boost::intrusive_ptr<cont_impl_type> cont_;
        bool future_obtained_;
    };
}}}

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace lcos
{
    // attach a local continuation to this future instance
    template <typename Result, typename RemoteResult>
    template <typename F>
//     inline typename detail::future_when_result<future<Result, RemoteResult>, F>::type
    inline future<
        typename boost::result_of<F(future<Result, RemoteResult>)>::type
    >
    future<Result, RemoteResult>::when(BOOST_FWD_REF(F) f)
    {
        typedef typename boost::result_of<F(future)>::type result_type;

        // create continuation
        typedef local::packaged_continuation<result_type, Result> cont_type;
        boost::shared_ptr<cont_type> p(
            boost::make_shared<cont_type>(
                util::bind(boost::forward<F>(f), util::placeholders::_1)
            )
        );

        // bind a on_completed handler to this future which will invoke the
        // continuation
        future_data_->set_on_completed(
            util::bind(&cont_type::on_value_ready, p, util::placeholders::_1));

        return p->get_future();
    }
}}

#endif
