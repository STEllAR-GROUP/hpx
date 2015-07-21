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
#include <hpx/util/unique_function.hpp>
#include <hpx/util/deferred_call.hpp>
#include <hpx/util/detail/value_or_error.hpp>

#include <boost/intrusive_ptr.hpp>
#include <boost/detail/atomic_count.hpp>
#include <boost/detail/scoped_enum_emulation.hpp>
#include <boost/thread/locks.hpp>

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

        virtual bool requires_delete()
        {
            return 0 == --count_;
        }

        virtual void destroy()
        {
            delete this;
        }

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
        if (p->requires_delete())
            p->destroy();
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
    class compose_cb_impl
    {
        HPX_MOVABLE_BUT_NOT_COPYABLE(compose_cb_impl);

    public:
        template <typename A1, typename A2>
        compose_cb_impl(A1 && f1, A2 && f2)
          : f1_(std::forward<A1>(f1))
          , f2_(std::forward<A2>(f2))
        {}

        compose_cb_impl(compose_cb_impl&& other)
          : f1_(std::move(other.f1_))
          , f2_(std::move(other.f2_))
        {}

        typedef void result_type;

        void operator()() const
        {
            f1_();
            f2_();
        }

        F1 f1_;
        F2 f2_;
    };

    template <typename F1, typename F2>
    static BOOST_FORCEINLINE util::unique_function_nonser<void()>
    compose_cb(F1 && f1, F2 && f2)
    {
        if (!f1)
            return std::forward<F2>(f2);
        else if (!f2)
            return std::forward<F1>(f1);

        // otherwise create a combined callback
        typedef compose_cb_impl<
            typename util::decay<F1>::type, typename util::decay<F2>::type
        > result_type;
        return result_type(std::forward<F1>(f1), std::forward<F2>(f2));
    }

    ///////////////////////////////////////////////////////////////////////////
    struct handle_continuation_recursion_count
    {
        handle_continuation_recursion_count()
          : count_(threads::get_continuation_recursion_count())
        {
            ++count_;
        }
        ~handle_continuation_recursion_count()
        {
            --count_;
        }

        std::size_t& count_;
    };

    ///////////////////////////////////////////////////////////////////////////
    HPX_EXPORT bool run_on_completed_on_new_thread(
        util::unique_function_nonser<bool()> && f, error_code& ec);

    ///////////////////////////////////////////////////////////////////////////
    template <typename Result>
    struct future_data : future_data_refcnt_base
    {
        typedef typename future_data_result<Result>::type result_type;
        typedef util::detail::value_or_error<result_type> data_type;
        typedef util::unique_function_nonser<void()> completed_callback_type;
        typedef lcos::local::spinlock mutex_type;

    public:
        future_data()
          : data_(), state_(empty)
        {}

        virtual void execute_deferred(error_code& ec = throws) {}

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

        // deferred execution of a given continuation
        bool run_on_completed(completed_callback_type && on_completed,
            boost::exception_ptr& ptr)
        {
            try {
                on_completed();
            }
            catch (...) {
                ptr = boost::current_exception();
                return false;
            }
            return true;
        }

        // make sure continuation invocation does not recurse deeper than
        // allowed
        void handle_on_completed(completed_callback_type && on_completed)
        {
            handle_continuation_recursion_count cnt;

            if (cnt.count_ <= HPX_CONTINUATION_MAX_RECURSION_DEPTH)
            {
                // directly execute continuation on this thread
                on_completed();
            }
            else
            {
                // re-spawn continuation on a new thread
                boost::intrusive_ptr<future_data> this_(this);

                error_code ec(lightweight);
                boost::exception_ptr ptr;
                if (!run_on_completed_on_new_thread(
                        util::deferred_call(&future_data::run_on_completed,
                            std::move(this_), std::move(on_completed),
                            std::ref(ptr)),
                        ec))
                {
                    // thread creation went wrong
                    if (ec) {
                        boost::unique_lock<mutex_type> l(this->mtx_);
                        set_result_locked(std::move(l),
                            hpx::detail::access_exception(ec));
                        return;
                    }

                    // re-throw exception in this context
                    HPX_ASSERT(ptr);        // exception should have been set
                    boost::rethrow_exception(ptr);
                }
            }
        }

        /// Set the result of the requested action.
        template <typename Target>
        void set_result(Target && data, error_code& ec = throws)
        {
            boost::unique_lock<mutex_type> l(this->mtx_);

            // check whether the data has already been set
            if (is_ready_locked()) {
                HPX_THROWS_IF(ec, promise_already_satisfied,
                    "future_data::set_result",
                    "data has already been set for this future");
                return;
            }

            return set_result_locked(std::move(l),
                std::forward<Target>(data), ec);
        }

        // Set result unconditionally
        template <typename Mutex, typename Target>
        void set_result_locked(boost::unique_lock<Mutex> l, Target && data,
            error_code& ec = throws)
        {
            completed_callback_type on_completed;

            on_completed = std::move(this->on_completed_);

            // set the data
            data_ = std::forward<Target>(data);

            // make sure the entry is full
            state_ = full;

            // handle all threads waiting for the future to become ready
            cond_.notify_all(std::move(l), ec);

            // Note: cv.notify_all() above 'consumes' the lock 'l' and leaves
            //       it unlocked when returning.

            // invoke the callback (continuation) function
            if (on_completed)
                handle_on_completed(std::move(on_completed));
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
                boost::unique_lock<mutex_type> l(this->mtx_);
                return set_result_locked(std::move(l),
                    boost::current_exception());
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
            boost::unique_lock<mutex_type> l(this->mtx_);
            state_ = empty;

            // release any stored data and callback functions
            data_ = data_type();
            on_completed_ = completed_callback_type();
        }

        // continuation support

        /// Set the callback which needs to be invoked when the future becomes
        /// ready. If the future is ready the function will be invoked
        /// immediately.
        void set_on_completed(completed_callback_type data_sink)
        {
            if (!data_sink) return;

            boost::unique_lock<mutex_type> l(this->mtx_);

            if (is_ready_locked()) {

                HPX_ASSERT(!on_completed_);

                // invoke the callback (continuation) function right away
                l.unlock();

                handle_on_completed(std::move(data_sink));
            }
            else {
                // store a combined callback wrapping the old and the new one
                this->on_completed_ = compose_cb(
                    std::move(data_sink), std::move(on_completed_));
            }
        }

        virtual void wait(error_code& ec = throws)
        {
            boost::unique_lock<mutex_type> l(mtx_);

            // block if this entry is empty
            if (state_ == empty) {
                cond_.wait(l, "future_data::wait", ec);
                if (ec) return;

                HPX_ASSERT(state_ != empty);
            }

            if (&ec != &throws)
                ec = make_success_code();
        }

        virtual BOOST_SCOPED_ENUM(future_status)
        wait_until(boost::chrono::steady_clock::time_point const& abs_time,
            error_code& ec = throws)
        {
            boost::unique_lock<mutex_type> l(mtx_);

            // block if this entry is empty
            if (state_ == empty) {
                threads::thread_state_ex_enum const reason =
                    cond_.wait_until(l, abs_time, "future_data::wait_until", ec);
                if (ec) return future_status::uninitialized;

                if (reason == threads::wait_signaled)
                    return future_status::timeout;

                HPX_ASSERT(state_ != empty);
                return future_status::ready;
            }

            if (&ec != &throws)
                ec = make_success_code();

            return future_status::ready; //-V110
        }

        /// Return whether or not the data is available for this
        /// \a future.
        bool is_ready() const
        {
            boost::unique_lock<mutex_type> l(mtx_);
            return is_ready_locked();
        }

        bool is_ready_locked() const
        {
            return state_ != empty;
        }

        bool has_value() const
        {
            boost::unique_lock<mutex_type> l(mtx_);
            return state_ != empty && data_.stores_value();
        }

        bool has_exception() const
        {
            boost::unique_lock<mutex_type> l(mtx_);
            return state_ != empty && data_.stores_error();
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
        timed_future_data(
            boost::chrono::steady_clock::time_point const& abs_time,
            Result_&& init)
        {
            boost::intrusive_ptr<timed_future_data> this_(this);

            error_code ec;
            threads::thread_id_type id = threads::register_thread_nullary(
                util::bind(util::one_shot(&timed_future_data::set_data),
                    std::move(this_), std::forward<Result_>(init)),
                "timed_future_data<Result>::timed_future_data",
                threads::suspended, true, threads::thread_priority_normal,
                std::size_t(-1), threads::thread_stacksize_default, ec);
            if (ec) {
                // thread creation failed, report error to the new future
                this->base_type::set_exception(hpx::detail::access_exception(ec));
            }

            // start new thread at given point in time
            threads::set_thread_state(id, abs_time, threads::pending,
                threads::wait_timeout, threads::thread_priority_boost, ec);
            if (ec) {
                // thread scheduling failed, report error to the new future
                this->base_type::set_exception(hpx::detail::access_exception(ec));
            }
        }

        void set_data(result_type const& value)
        {
            this->base_type::set_result(value);
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
            boost::lock_guard<mutex_type> l(this->mtx_);
            return id_;
        }
        void set_id(threads::thread_id_type id)
        {
            boost::lock_guard<mutex_type> l(this->mtx_);
            id_ = id;
        }

    public:
        task_base()
          : started_(false), id_(threads::invalid_thread_id), sched_(0)
        {}

        task_base(threads::executor& sched)
          : started_(false), id_(threads::invalid_thread_id),
            sched_(sched ? &sched : 0)
        {}

        virtual void execute_deferred(error_code& ec = throws)
        {
            if (!started_test_and_set())
                this->do_run();
        }

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
        wait_until(boost::chrono::steady_clock::time_point const& abs_time,
            error_code& ec = throws)
        {
            if (!started_test())
                return future_status::deferred; //-V110
            else
                return this->future_data<Result>::wait_until(abs_time, ec);
        };

    private:
        bool started_test() const
        {
            boost::lock_guard<mutex_type> l(this->mtx_);
            return started_;
        }

        bool started_test_and_set()
        {
            boost::lock_guard<mutex_type> l(this->mtx_);
            if (started_)
                return true;

            started_ = true;
            return false;
        }

        void check_started()
        {
            boost::lock_guard<mutex_type> l(this->mtx_);
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
        void apply(BOOST_SCOPED_ENUM(launch) policy,
            threads::thread_priority priority,
            threads::thread_stacksize stacksize, error_code& ec)
        {
            check_started();

            future_base_type this_(this);

            char const* desc = hpx::threads::get_thread_description(
                hpx::threads::get_self_id());

            if (sched_) {
                sched_->add(util::bind(&task_base::run_impl, std::move(this_)),
                    desc ? desc : "task_base::apply", threads::pending, false,
                    stacksize, ec);
            }
            else if (policy == launch::fork) {
                threads::register_thread_plain(
                    util::bind(&task_base::run_impl, std::move(this_)),
                    desc ? desc : "task_base::apply", threads::pending, false,
                    threads::thread_priority_boost, get_worker_thread_num(),
                    stacksize, ec);
            }
            else {
                threads::register_thread_plain(
                    util::bind(&task_base::run_impl, std::move(this_)),
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
            boost::unique_lock<mutex_type> l(this->mtx_);
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
