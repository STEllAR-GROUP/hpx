//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_LCOS_DETAIL_FUTURE_DATA_MAR_06_2012_1055AM)
#define HPX_LCOS_DETAIL_FUTURE_DATA_MAR_06_2012_1055AM

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/threads/thread_helpers.hpp>
#include <hpx/util/full_empty_memory.hpp>
#include <hpx/util/unused.hpp>
#include <hpx/util/value_or_error.hpp>
#include <hpx/util/unlock_lock.hpp>

#include <boost/move/move.hpp>
#include <boost/mpl/if.hpp>
#include <boost/type_traits/is_same.hpp>
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
namespace hpx { namespace lcos { namespace detail
{
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

    private:
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
    template <typename Result, typename RemoteResult =
        typename traits::promise_remote_result<Result>::type>
    struct future_data_base : future_data_refcnt_base
    {
        typedef typename boost::mpl::if_<
            boost::is_same<void, Result>, util::unused_type, Result
        >::type result_type;
        typedef
            HPX_STD_FUNCTION<void(future<Result, RemoteResult>)>
        completed_callback_type;
        typedef lcos::local::spinlock mutex_type;

        virtual void deleting_owner() {}

        virtual result_type get_data(error_code& ec = throws) = 0;
        virtual result_type move_data(error_code& ec = throws) = 0;
        virtual bool is_ready() const = 0;
        virtual bool has_value() const = 0;
        virtual bool has_exception() const = 0;
        virtual BOOST_SCOPED_ENUM(future_status) get_state() const = 0;

        // cancellation is disabled by default
        virtual bool is_cancelable() const
        {
            return false;
        }
        virtual void cancel()
        {
            HPX_THROW_EXCEPTION(future_does_not_support_cancellation,
                "future_data_base::cancel",
                "this future does not support cancellation");
        }

        // continuation support
        virtual completed_callback_type
            set_on_completed(BOOST_RV_REF(completed_callback_type)) = 0;
        virtual completed_callback_type
            set_on_completed_locked(BOOST_RV_REF(completed_callback_type)) = 0;
        virtual void reset_on_completed() = 0;

        // wait support
        void wake_me_up(threads::thread_id_type id)
        {
            threads::set_thread_state(id, threads::pending, threads::wait_timeout);
        }

        struct reset_cb
        {
            template <typename F>
            reset_cb(future_data_base& fb, BOOST_FWD_REF(F) f)
              : target_(fb),
                oldcb_(fb.set_on_completed_locked(boost::forward<F>(f)))
            {}
            ~reset_cb()
            {
                target_.set_on_completed_locked(boost::move(oldcb_));
            }

            future_data_base& target_;
            completed_callback_type oldcb_;
        };

        void wait()
        {
            mutex_type::scoped_lock l(mtx_);
            if (!is_ready()) {
                boost::intrusive_ptr<future_data_base> this_(this);
                reset_cb r(*this, util::bind(
                    &future_data_base::wake_me_up, this_, threads::get_self_id()));

                util::unlock_the_lock<mutex_type::scoped_lock> ul(l);
                this_thread::suspend(threads::suspended);
            }
        }

        BOOST_SCOPED_ENUM(future_status)
        wait_for(boost::posix_time::time_duration const& p)
        {
            mutex_type::scoped_lock l(mtx_);
            if (!is_ready()) {
                boost::intrusive_ptr<future_data_base> this_(this);
                reset_cb r(*this, util::bind(
                    &future_data_base::wake_me_up, this_, threads::get_self_id()));

                util::unlock_the_lock<mutex_type::scoped_lock> ul(l);
                return (this_thread::suspend(p) == threads::wait_signaled) ?
                    future_status::ready : future_status::timeout;
            }
            return future_status::ready;
        }

        BOOST_SCOPED_ENUM(future_status)
        wait_until(boost::posix_time::ptime const& at)
        {
            mutex_type::scoped_lock l(mtx_);
            if (!is_ready()) {
                boost::intrusive_ptr<future_data_base> this_(this);
                reset_cb r(*this, util::bind(
                    &future_data_base::wake_me_up, this_, threads::get_self_id()));

                util::unlock_the_lock<mutex_type::scoped_lock> ul(l);
                return (this_thread::suspend(at) == threads::wait_signaled) ?
                    future_status::ready : future_status::timeout;
            }
            return future_status::ready;
        }

    protected:
        mutable mutex_type mtx_;

    protected:
        future_data_base() {}
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename Result, typename RemoteResult =
        typename traits::promise_remote_result<Result>::type>
    struct future_data : future_data_base<Result, RemoteResult>
    {
    public:
        typedef future_data_base<Result, RemoteResult> base_type;
        typedef typename base_type::result_type result_type;
        typedef typename base_type::mutex_type mutex_type;
        typedef boost::exception_ptr error_type;
        typedef util::value_or_error<result_type> data_type;

        typedef typename base_type::completed_callback_type
            completed_callback_type;

        template <typename R, typename RR> friend class future;

    protected:
        future_data() {}

        future_data(completed_callback_type const& data_sink)
          : on_completed_(data_sink)
        {}

    public:
        result_type handle_error(data_type const& d, error_code &ec)
        {
            // an error has been reported in the meantime, throw or set
            // the error code
            if (&ec == &throws) {
                boost::rethrow_exception(d.get_error());
                // never reached
            }
            else {
                try {
                    boost::rethrow_exception(d.get_error());
                }
                catch (hpx::exception const& he) {
                    ec = make_error_code(he.get_error(), he.what(),
                        hpx::rethrow);
                }
            }
            return result_type();
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
        result_type get_data(error_code& ec = throws)
        {
            // yields control if needed
            data_type d;
            data_.read(d, ec);      // copies the data out of the store
            if (ec) return result_type();

            // the thread has been re-activated by one of the actions
            // supported by this promise (see \a promise::set_event
            // and promise::set_exception).
            if (!d.stores_value())
                return handle_error(d, ec);

            // no error has been reported, return the result
            return boost::move(d.get_value());
        }

        result_type move_data(error_code& ec = throws)
        {
            // yields control if needed
            data_type d;
            data_.read_and_empty(d, ec); // moves the data from the store
            if (ec) return result_type();

            // the thread has been re-activated by one of the actions
            // supported by this promise (see \a promise::set_event
            // and promise::set_exception).
            if (!d.stores_value())
                return handle_error(d, ec);

            // no error has been reported, return the result
            return boost::move(d.get_value());
        }

        // helper functions for setting data (if successful) or the error (if
        // non-successful)
        template <typename T>
        void set_data(BOOST_FWD_REF(T) result)
        {
            // set the received result, reset error status
            try {
                // check weather the data already has been set
                if (!data_.is_empty()) {
                    HPX_THROW_EXCEPTION(future_already_satisfied,
                        "packaged_task::set_data<Result>",
                        "data has already been set for this future");
                }

               typedef typename boost::remove_const<
                    typename hpx::util::detail::remove_reference<T>::type
                >::type naked_type;

                typedef traits::get_remote_result<
                    result_type, naked_type
                > get_remote_result_type;

                // store the value
                typename mutex_type::scoped_lock l(this->mtx_);
                if (!on_completed_.empty()) {
                    // this future instance coincidentally keeps us alive
                    future<Result, RemoteResult> f(this);
                    data_.set(boost::move(get_remote_result_type::call(
                        boost::forward<T>(result))));

                    // invoke the callback (continuation) function
                    on_completed_(f);
                    on_completed_.reset();
                }
                else {
                  data_.set(boost::move(get_remote_result_type::call(
                        boost::forward<T>(result))));
                }
            }
            catch (hpx::exception const&) {
                // store the error instead
                set_exception(boost::current_exception());
            }
        }

        // trigger the future with the given error condition
        void set_exception(boost::exception_ptr const& e)
        {
            // store the error code
            typename mutex_type::scoped_lock l(this->mtx_);
            if (!on_completed_.empty()) {
                // this future coincidentally instance keeps us alive
                future<Result, RemoteResult> f(this);
                data_.set(e);

                // invoke the callback (continuation) function
                on_completed_(f);
                on_completed_.reset();
            }
            else {
                data_.set(e);
            }
        }

        void set_error(error e, char const* f, char const* msg)
        {
            try {
                HPX_THROW_EXCEPTION(e, f, msg);
            }
            catch (hpx::exception const&) {
                // store the error code
                set_exception(boost::current_exception());
            }
        }

        /// Return whether or not the data is available for this
        /// \a promise.
        bool is_ready() const
        {
            return !data_.is_empty();
        }

    private:
        static bool has_data_helper(data_type const& d)
        {
            return d.stores_value();
        }

    public:
        bool has_value() const
        {
            return is_ready() && data_.peek(&has_data_helper);
        }

        bool has_exception() const
        {
            return is_ready() && !data_.peek(&has_data_helper);
        }

        BOOST_SCOPED_ENUM(future_status) get_state() const
        {
            return is_ready() ? future_status::ready : future_status::deferred;
        }

        /// Reset the promise to allow to restart an asynchronous
        /// operation. Allows any subsequent set_data operation to succeed.
        void reset(error_code& ec = throws)
        {
            data_.set_empty(ec);
        }

        /// Set the callback which needs to be invoked when the future becomes
        /// ready. If the future is ready the function will be invoked
        /// immediately.
        completed_callback_type
        set_on_completed(BOOST_RV_REF(completed_callback_type) data_sink)
        {
            typename mutex_type::scoped_lock l(this->mtx_);
            return boost::move(set_on_completed_locked(boost::move(data_sink)));
        }

        completed_callback_type
        set_on_completed_locked(BOOST_RV_REF(completed_callback_type) data_sink)
        {
            completed_callback_type retval = boost::move(on_completed_);
            on_completed_ = boost::move(data_sink);
            if (!on_completed_.empty() && this->is_ready()) {
                // this future coincidentally instance keeps us alive
                future<Result, RemoteResult> f(this);

                // invoke the callback (continuation) function
                on_completed_(f);
                on_completed_.reset();
            }
            return retval;
        }

        void reset_on_completed()
        {
            completed_callback_type data_sink;
            {
                typename mutex_type::scoped_lock l(this->mtx_);
                std::swap(on_completed_, data_sink);
            }
        }

    private:
        util::full_empty<data_type> data_;
        completed_callback_type on_completed_;
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
          : started_(false), id_(threads::invalid_thread_id)
        {}

        // retrieving the value
        result_type get(error_code& ec = throws)
        {
            if (!started_)
                run(ec);

            return boost::move(this->get_data(ec));
        }

        // moving out the value
        result_type move(error_code& ec = throws)
        {
            if (!started_)
                run(ec);

            return boost::move(this->move_data(ec));
        }

    private:
        void check_started()
        {
            typename mutex_type::scoped_lock l(this->mtx_);
            if (started_) {
                HPX_THROW_EXCEPTION(task_already_started,
                    "task_base::run", "this task has already been started");
                return;
            }
            started_ = true;
        }

    public:
        // run synchronously
        void run()
        {
            check_started();
            this->do_run();
        }

        // run in a separate thread
        void apply()
        {
            check_started();
            future_base_type this_(this);
            applier::register_thread_plain(
                HPX_STD_BIND(&task_base::run_impl, this_),
                "task_base::apply");
        }

    private:
        struct reset_id
        {
            reset_id(task_base& target)
              : target_(target)
            {
                target.set_id(threads::get_self().get_thread_id());
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
        void set_data(BOOST_FWD_REF(T) result)
        {
            started_ = true;
            this->future_data<Result>::set_data(boost::forward<T>(result));
        }

        void set_exception(boost::exception_ptr const& e)
        {
            started_ = true;
            this->future_data<Result>::set_exception(e);
        }

        virtual void do_run() = 0;

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
                        "task_base<Result>::cancel",
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
    };
}}}

#endif
