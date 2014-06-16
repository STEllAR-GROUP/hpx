//  Copyright (c) 2007-2013 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_LCOS_DETAIL_FUTURE_DATA_MAR_06_2012_1055AM)
#define HPX_LCOS_DETAIL_FUTURE_DATA_MAR_06_2012_1055AM

#include <hpx/hpx_fwd.hpp>
#include <hpx/lcos/local/detail/condition_variable.hpp>
#include <hpx/lcos/local/spinlock.hpp>
#include <hpx/traits/get_remote_result.hpp>
#include <hpx/runtime/threads/thread_helpers.hpp>
#include <hpx/runtime/threads/thread_executor.hpp>
#include <hpx/util/bind.hpp>
#include <hpx/util/decay.hpp>
#include <hpx/util/move.hpp>
#include <hpx/util/unused.hpp>
#include <hpx/util/detail/value_or_error.hpp>

#include <boost/intrusive_ptr.hpp>
#include <boost/detail/atomic_count.hpp>
#include <boost/detail/scoped_enum_emulation.hpp>
#include <boost/preprocessor/cat.hpp>
#include <boost/preprocessor/repetition/enum.hpp>
#include <boost/preprocessor/repetition/repeat_from_to.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace lcos
{
    BOOST_SCOPED_ENUM_START(future_status)
    {
        ready, timeout, deferred, uninitialized
    };
    BOOST_SCOPED_ENUM_END
}}

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace lcos
{

namespace local { template <typename T> struct channel; }

namespace detail
{
    template <typename Result> struct future_data;

    ///////////////////////////////////////////////////////////////////////
    struct future_data_refcnt_base;

    void intrusive_ptr_add_ref(future_data_refcnt_base* p);
    void intrusive_ptr_release(future_data_refcnt_base* p);

    ///////////////////////////////////////////////////////////////////////
    struct future_data_refcnt_base
    {
    public:
        typedef void has_future_data_refcnt_base;

        virtual ~future_data_refcnt_base() {}

    protected:
        future_data_refcnt_base() : count_(0) {}

        // reference counting
        friend void intrusive_ptr_add_ref(future_data_refcnt_base* p);
        friend void intrusive_ptr_release(future_data_refcnt_base* p);

        boost::detail::atomic_count count_;
    };

    /// support functions for boost::intrusive_ptr
    inline void intrusive_ptr_add_ref(future_data_refcnt_base* p)
    {
        ++p->count_;
    }
    inline void intrusive_ptr_release(future_data_refcnt_base* p)
    {
        if (0 == --p->count_)
            delete p;
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename Result>
    struct future_data_result
    {
        typedef Result type;
    };

    template <>
    struct future_data_result<void>
    {
        typedef util::unused_type type;
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename F1, typename F2>
    struct compose_cb_impl
    {
        template <typename A1, typename A2>
        compose_cb_impl(A1 && f1, A2 && f2)
          : f1_(std::forward<A1>(f1))
          , f2_(std::forward<A2>(f2))
        {}

        typedef void result_type;

        void operator()() const
        {
            f1_();
            f2_();
        }

        typename boost::remove_reference<F1>::type f1_;
        typename boost::remove_reference<F2>::type f2_;
    };

    template <typename F1, typename F2>
    static BOOST_FORCEINLINE HPX_STD_FUNCTION<void()>
    compose_cb(F1 && f1, F2 && f2)
    {
        if (f1.empty())
            return std::forward<F2>(f2);
        else if (f2.empty())
            return std::forward<F1>(f1);

        // otherwise create a combined callback
        return compose_cb_impl<F1, F2>(
            std::forward<F1>(f1), std::forward<F2>(f2));
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename Result>
    struct future_data : future_data_refcnt_base
    {
        typedef typename future_data_result<Result>::type result_type;
        typedef util::detail::value_or_error<result_type> data_type;
        typedef HPX_STD_FUNCTION<void()> completed_callback_type;
        typedef lcos::local::spinlock mutex_type;

    public:
        future_data()
          : data_(), state_(empty)
        {}

        virtual void deleting_owner() {}

        // cancellation is disabled by default
        virtual bool cancelable() const
        {
            return false;
        }
        virtual void cancel()
        {
            HPX_THROW_EXCEPTION(future_does_not_support_cancellation,
                "future_data::cancel",
                "this future does not support cancellation");
        }

        /// Get the result of the requested action. This call blocks (yields
        /// control) if the result is not ready. As soon as the result has been
        /// returned and the waiting thread has been re-scheduled by the thread
        /// manager the function will return.
        ///
        /// \param ec     [in,out] this represents the error status on exit,
        ///               if this is pre-initialized to \a hpx#throws
        ///               the function will throw on error instead. If the
        ///               operation blocks and is aborted because the object
        ///               went out of scope, the code \a hpx#yield_aborted is
        ///               set or thrown.
        ///
        /// \note         If there has been an error reported (using the action
        ///               \a base_lco#set_exception), this function will throw an
        ///               exception encapsulating the reported error code and
        ///               error description if <code>&ec == &throws</code>.
        virtual data_type& get_result(error_code& ec = throws)
        {
            // yields control if needed
            wait(ec);
            if (ec) return data_;

            if (data_.is_empty()) {
                // the value has already been moved out of this future
                HPX_THROWS_IF(ec, no_state,
                    "future_data::get_result",
                    "this future has no valid shared state");
                return data_;
            }

            // the thread has been re-activated by one of the actions
            // supported by this promise (see \a promise::set_event
            // and promise::set_exception).
            if (data_.stores_error())
            {
                // an error has been reported in the meantime, throw or set
                // the error code
                if (&ec == &throws) {
                    boost::rethrow_exception(data_.get_error());
                    // never reached
                }
                else {
                    ec = make_error_code(data_.get_error());
                }
            }
            return data_;
        }

        /// Set the result of the requested action.
        template <typename Target>
        void set_result(Target && data, error_code& ec = throws)
        {
            completed_callback_type on_completed;
            {
                typename mutex_type::scoped_lock l(this->mtx_);

                // check whether the data already has been set
                if (is_ready_locked()) {
                    HPX_THROWS_IF(ec, promise_already_satisfied,
                        "future_data::set_result",
                        "data has already been set for this future");
                    return;
                }

                on_completed = std::move(this->on_completed_);

                // set the data
                data_ = std::forward<Target>(data);

                // make sure the entry is full
                state_ = full;

                // handle all threads waiting for the block to become full
                cond_.notify_all(l, ec);
            }

            // invoke the callback (continuation) function
            if (!on_completed.empty())
                on_completed();
        }

        // helper functions for setting data (if successful) or the error (if
        // non-successful)
        template <typename T>
        void set_data(T && result)
        {
            // set the received result, reset error status
            try {
               typedef typename util::decay<T>::type naked_type;

                typedef traits::get_remote_result<
                    result_type, naked_type
                > get_remote_result_type;

                // store the value
                set_result(std::move(get_remote_result_type::call(
                        std::forward<T>(result))));
            }
            catch (hpx::exception const&) {
                // store the error instead
                set_result(boost::current_exception());
            }
        }

        // trigger the future with the given error condition
        void set_exception(boost::exception_ptr const& e)
        {
            // store the error code
            set_result(e);
        }

        void set_error(error e, char const* f, char const* msg)
        {
            try {
                HPX_THROW_EXCEPTION(e, f, msg);
            }
            catch (hpx::exception const&) {
                // store the error code
                set_result(boost::current_exception());
            }
        }

        /// Reset the promise to allow to restart an asynchronous
        /// operation. Allows any subsequent set_data operation to succeed.
        void reset(error_code& /*ec*/ = throws)
        {
            typename mutex_type::scoped_lock l(this->mtx_);
            state_ = empty;

            // release any stored data and callback functions
            data_ = data_type();
            on_completed_ = completed_callback_type();
        }

        // continuation support

        /// Set the callback which needs to be invoked when the future becomes
        /// ready. If the future is ready the function will be invoked
        /// immediately.
        completed_callback_type
        set_on_completed(completed_callback_type data_sink)
        {
            typename mutex_type::scoped_lock l(this->mtx_);

            completed_callback_type retval = std::move(this->on_completed_);

            if (!data_sink.empty() && is_ready_locked()) {
                // invoke the callback (continuation) function right away
                l.unlock();

                if (!retval.empty())
                    retval();
                data_sink();
            }
            else if (!retval.empty()) {
                // store a combined callback wrapping the old and the new one
                this->on_completed_ = std::move(
                    compose_cb(std::move(data_sink), retval));

                l.unlock();
            }
            else {
                // store the new callback
                this->on_completed_ = std::move(data_sink);

                l.unlock();
            }

            return std::move(retval);
        }

        completed_callback_type reset_on_completed_locked()
        {
            return std::move(this->on_completed_);
        }

        virtual void wait(error_code& ec = throws)
        {
            typename mutex_type::scoped_lock l(mtx_);

            // block if this entry is empty
            if (state_ == empty) {
                cond_.wait(l, "future_data::wait", ec);
                if (ec) return;
            }

            if (&ec != &throws)
                ec = make_success_code();
        }

        virtual BOOST_SCOPED_ENUM(future_status)
        wait_for(boost::posix_time::time_duration const& p, error_code& ec = throws)
        {
            typename mutex_type::scoped_lock l(mtx_);

            // block if this entry is empty
            if (state_ == empty) {
                threads::thread_state_ex_enum const reason =
                    cond_.wait_for(l, p, "future_data::wait_for", ec);
                if (ec) return future_status::uninitialized;

                return (reason == threads::wait_signaled) ?
                    future_status::timeout : future_status::ready; //-V110
            }

            if (&ec != &throws)
                ec = make_success_code();

            return future_status::ready; //-V110
        }

        virtual BOOST_SCOPED_ENUM(future_status)
        wait_until(boost::posix_time::ptime const& at, error_code& ec = throws)
        {
            typename mutex_type::scoped_lock l(mtx_);

            // block if this entry is empty
            if (state_ == empty) {
                threads::thread_state_ex_enum const reason =
                    cond_.wait_until(l, at, "future_data::wait_until", ec);
                if (ec) return future_status::uninitialized;

                return (reason == threads::wait_signaled) ?
                    future_status::timeout : future_status::ready; //-V110
            }

            if (&ec != &throws)
                ec = make_success_code();

            return future_status::ready; //-V110
        }

        /// Return whether or not the data is available for this
        /// \a future.
        bool is_ready() const
        {
            typename mutex_type::scoped_lock l(mtx_);
            return is_ready_locked();
        }

        bool is_ready_locked() const
        {
            return state_ != empty;
        }

        bool has_value() const
        {
            typename mutex_type::scoped_lock l(mtx_);
            return state_ != empty && data_.stores_value();
        }

        bool has_exception() const
        {
            typename mutex_type::scoped_lock l(mtx_);
            return state_ != empty && data_.stores_error();
        }

        BOOST_SCOPED_ENUM(future_status) get_status() const
        {
            typename mutex_type::scoped_lock l(mtx_);
            return state_ != empty ? future_status::ready : future_status::deferred; //-V110
        }

    protected:
        mutable mutex_type mtx_;
        data_type data_;                            // protected data
        completed_callback_type on_completed_;

    private:
        local::detail::condition_variable cond_;    // threads waiting in read
        full_empty_state state_;                    // current full/empty state
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename Result>
    struct timed_future_data : future_data<Result>
    {
    public:
        typedef future_data<Result> base_type;
        typedef typename base_type::result_type result_type;
        typedef typename base_type::mutex_type mutex_type;
        typedef typename base_type::data_type data_type;

    public:
        timed_future_data() {}

        template <typename Result_>
        timed_future_data(boost::posix_time::ptime const& at,
            Result_ && init)
        {
            at_time(at, std::forward<Result_>(init));
        }

        template <typename Result_>
        timed_future_data(boost::posix_time::time_duration const& d,
            Result_ && init)
        {
            at_time(d, std::forward<Result_>(init));
        }

        void set_data(result_type const& value)
        {
            this->base_type::set_result(value);
        }

        template <typename TimeSpec, typename Result_>
        void at_time(TimeSpec const& tpoint, Result_ && init)
        {
            boost::intrusive_ptr<timed_future_data> this_(this);

            error_code ec;
            threads::thread_id_type id = threads::register_thread_nullary(
                HPX_STD_BIND(&timed_future_data::set_data, this_,
                    std::forward<Result_>(init)),
                "timed_future_data<Result>::timed_future_data",
                threads::suspended, true, threads::thread_priority_normal,
                std::size_t(-1), threads::thread_stacksize_default, ec);
            if (ec) {
                // thread creation failed, report error to the new future
                this->base_type::set_exception(hpx::detail::access_exception(ec));
            }

            // start new thread at given point in time
            threads::set_thread_state(id, tpoint, threads::pending,
                threads::wait_timeout, threads::thread_priority_critical, ec);
            if (ec) {
                // thread scheduling failed, report error to the new future
                this->base_type::set_exception(hpx::detail::access_exception(ec));
            }
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename Result>
    struct task_base : future_data<Result>
    {
    private:
        typedef typename future_data<Result>::mutex_type mutex_type;
        typedef boost::intrusive_ptr<task_base> future_base_type;

    protected:
        typedef typename future_data<Result>::result_type result_type;
        typedef typename future_data<Result>::data_type data_type;

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

    public:
        task_base()
          : started_(false), id_(threads::invalid_thread_id), sched_(0)
        {}

        task_base(threads::executor& sched)
          : started_(false), id_(threads::invalid_thread_id), sched_(&sched)
        {}

        // retrieving the value
        virtual data_type& get_result(error_code& ec = throws)
        {
            if (!started_test_and_set())
                this->do_run();
            return this->future_data<Result>::get_result(ec);
        }

        // wait support
        virtual void wait(error_code& ec = throws)
        {
            if (!started_test_and_set())
                this->do_run();
            else
                this->future_data<Result>::wait(ec);
        }

        virtual BOOST_SCOPED_ENUM(future_status)
        wait_for(boost::posix_time::time_duration const& /*p*/, error_code& /*ec*/ = throws)
        {
            return future_status::deferred; //-V110
        }

        virtual BOOST_SCOPED_ENUM(future_status)
        wait_until(boost::posix_time::ptime const& /*at*/, error_code& /*ec*/ = throws)
        {
            return future_status::deferred; //-V110
        };

    private:
        bool started_test_and_set()
        {
            typename mutex_type::scoped_lock l(this->mtx_);
            if (started_)
                return true;

            started_ = true;
            return false;
        }

        void check_started()
        {
            typename mutex_type::scoped_lock l(this->mtx_);
            if (started_) {
                HPX_THROW_EXCEPTION(task_already_started,
                    "task_base::check_started",
                    "this task has already been started");
                return;
            }
            started_ = true;
        }

    public:
        // run synchronously
        void run()
        {
            check_started();
            this->do_run();       // always on this thread
        }

        // run in a separate thread
        void apply(threads::thread_priority priority,
            threads::thread_stacksize stacksize, error_code& ec)
        {
            check_started();

            future_base_type this_(this);

            char const* desc = hpx::threads::get_thread_description(
                hpx::threads::get_self_id());

            if (sched_) {
                sched_->add(HPX_STD_BIND(&task_base::run_impl, this_),
                    desc ? desc : "task_base::apply", threads::pending, false,
                    stacksize, ec);
            }
            else {
                threads::register_thread_plain(
                    HPX_STD_BIND(&task_base::run_impl, this_),
                    desc ? desc : "task_base::apply", threads::pending, false,
                    priority, std::size_t(-1), stacksize, ec);
            }
        }

    private:
        struct reset_id
        {
            reset_id(task_base& target)
              : target_(target)
            {
                target.set_id(threads::get_self_id());
            }
            ~reset_id()
            {
                target_.set_id(threads::invalid_thread_id);
            }
            task_base& target_;
        };

    protected:
        threads::thread_state_enum run_impl()
        {
            reset_id r(*this);
            this->do_run();
            return threads::terminated;
        }

    public:
        void deleting_owner()
        {
            typename mutex_type::scoped_lock l(this->mtx_);
            if (!started_) {
                started_ = true;
                l.unlock();
                this->set_error(broken_task, "task_base::deleting_owner",
                    "deleting task owner before future has become ready");
            }
        }

        template <typename T>
        void set_data(T && result)
        {
            HPX_ASSERT(started_);
            this->future_data<Result>::set_result(std::forward<T>(result));
        }

        void set_exception(boost::exception_ptr const& e)
        {
            HPX_ASSERT(started_);
            this->future_data<Result>::set_exception(e);
        }

        virtual void do_run() = 0;

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
                        "task_base<Result>::cancel",
                        "future has been canceled");
                }
                else {
                    HPX_THROW_EXCEPTION(future_can_not_be_cancelled,
                        "task_base<Result>::cancel",
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
        threads::executor* sched_;
    };
}}}

#endif
