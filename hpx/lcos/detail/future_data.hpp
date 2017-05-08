//  Copyright (c) 2007-2017 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_LCOS_DETAIL_FUTURE_DATA_MAR_06_2012_1055AM)
#define HPX_LCOS_DETAIL_FUTURE_DATA_MAR_06_2012_1055AM

#include <hpx/config.hpp>
#include <hpx/error_code.hpp>
#include <hpx/lcos/local/detail/condition_variable.hpp>
#include <hpx/lcos/local/spinlock.hpp>
#include <hpx/runtime/get_worker_thread_num.hpp>
#include <hpx/runtime/launch_policy.hpp>
#include <hpx/runtime/threads/thread_executor.hpp>
#include <hpx/runtime/threads/thread_helpers.hpp>
#include <hpx/runtime/threads/coroutines/detail/get_stack_pointer.hpp>
#include <hpx/throw_exception.hpp>
#include <hpx/traits/future_access.hpp>
#include <hpx/traits/get_remote_result.hpp>
#include <hpx/util/annotated_function.hpp>
#include <hpx/util/assert_owns_lock.hpp>
#include <hpx/util/atomic_count.hpp>
#include <hpx/util/bind.hpp>
#include <hpx/util/decay.hpp>
#include <hpx/util/deferred_call.hpp>
#include <hpx/util/steady_clock.hpp>
#include <hpx/util/unique_function.hpp>
#include <hpx/util/unused.hpp>

#include <boost/exception_ptr.hpp>
#include <boost/intrusive_ptr.hpp>

#include <chrono>
#include <cstddef>
#include <functional>
#include <memory>
#include <mutex>
#include <type_traits>
#include <utility>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace lcos
{
    enum class future_status
    {
        ready, timeout, deferred, uninitialized
    };
}}

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace lcos
{

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
    private:
        typedef util::unique_function_nonser<void()> completed_callback_type;

    public:
        typedef void has_future_data_refcnt_base;

        virtual ~future_data_refcnt_base() {}

        virtual void set_on_completed(completed_callback_type) = 0;

        virtual bool requires_delete()
        {
            return 0 == --count_;
        }

        virtual void destroy()
        {
            delete this;
        }

        // This is a tag type used to convey the information that the caller is
        // _not_ going to addref the future_data instance
        struct init_no_addref {};

    protected:
        future_data_refcnt_base() : count_(0) {}
        future_data_refcnt_base(init_no_addref) : count_(1) {}

        // reference counting
        friend void intrusive_ptr_add_ref(future_data_refcnt_base* p);
        friend void intrusive_ptr_release(future_data_refcnt_base* p);

        util::atomic_count count_;
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

        template <typename U>
        HPX_FORCEINLINE static
        U && set(U && u)
        {
            return std::forward<U>(u);
        }
    };

    template <typename Result>
    struct future_data_result<Result&>
    {
        typedef Result* type;

        HPX_FORCEINLINE static
        Result* set(Result* u)
        {
            return u;
        }

        HPX_FORCEINLINE static
        Result* set(Result& u)
        {
            return &u;
        }
    };

    template <>
    struct future_data_result<void>
    {
        typedef util::unused_type type;

        HPX_FORCEINLINE static
        util::unused_type set(util::unused_type u)
        {
            return u;
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename R>
    struct future_data_storage
    {
        typedef typename future_data_result<R>::type value_type;
        typedef boost::exception_ptr error_type;

        // determine the required alignment, define aligned storage of proper
        // size
        HPX_STATIC_CONSTEXPR std::size_t max_alignment =
            (std::alignment_of<value_type>::value >
             std::alignment_of<error_type>::value) ?
            std::alignment_of<value_type>::value
          : std::alignment_of<error_type>::value;

        HPX_STATIC_CONSTEXPR std::size_t max_size =
                (sizeof(value_type) > sizeof(error_type)) ?
                    sizeof(value_type) : sizeof(error_type);

        typedef typename std::aligned_storage<max_size, max_alignment>::type type;
    };

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
    template <typename F1, typename F2>
    class compose_cb_impl
    {
        HPX_MOVABLE_ONLY(compose_cb_impl);

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

        void operator()()
        {
            bool recurse_asynchronously = hpx::threads::get_self_ptr() == nullptr;
#if defined(HPX_HAVE_THREADS_GET_STACK_POINTER)
            recurse_asynchronously =
                !this_thread::has_sufficient_stack_space();
#else
            handle_continuation_recursion_count cnt;
            recurse_asynchronously = recurse_asynchronously ||
                cnt.count_ > HPX_CONTINUATION_MAX_RECURSION_DEPTH;
#endif
            if (recurse_asynchronously)
            {
                error_code ec;
                threads::thread_id_type id = threads::register_thread_nullary(
                    compose_cb_impl(std::move(f1_), std::move(f2_)),
                    "compose_cb",
                    threads::pending, true, threads::thread_priority_boost,
                    std::size_t(-1), threads::thread_stacksize_current, ec);
                return;
            }

            f1_();
            f2_();
        }

    private:
        F1 f1_;
        F2 f2_;
    };

    template <typename F1, typename F2>
    static HPX_FORCEINLINE util::unique_function_nonser<void()>
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
    HPX_EXPORT bool run_on_completed_on_new_thread(
        util::unique_function_nonser<bool()> && f, error_code& ec);

    ///////////////////////////////////////////////////////////////////////////
    template <typename Result>
    struct future_data;

    template <>
    struct future_data<traits::detail::future_data_void> : future_data_refcnt_base
    {
        future_data()
          : state_(empty)
        {}

        future_data(init_no_addref no_addref)
          : future_data_refcnt_base(no_addref), state_(empty)
        {}

        typedef lcos::local::spinlock mutex_type;
        typedef util::unused_type result_type;
        typedef future_data_refcnt_base::init_no_addref init_no_addref;

        virtual ~future_data() HPX_NOEXCEPT {}
        virtual void execute_deferred(error_code& = throws) = 0;
        virtual bool cancelable() const = 0;
        virtual void cancel() = 0;
        virtual result_type* get_result_void(error_code& = throws) = 0;
        virtual void wait(error_code& = throws) = 0;
        virtual future_status wait_until(util::steady_clock::time_point const&,
            error_code& = throws) = 0;

        enum state
        {
            empty = 0,
            ready = 1,
            value = 2 | ready,
            exception = 4 | ready
        };

        /// Return whether or not the data is available for this
        /// \a future.
        bool is_ready() const
        {
            std::unique_lock<mutex_type> l(mtx_);
            return is_ready_locked(l);
        }

        template <typename Lock>
        bool is_ready_locked(Lock& l) const
        {
            HPX_ASSERT_OWNS_LOCK(l);
            return (state_ & ready) != 0;
        }

        bool has_value() const
        {
            std::unique_lock<mutex_type> l(mtx_);
            return state_ == value;
        }

        bool has_exception() const
        {
            std::unique_lock<mutex_type> l(mtx_);
            return state_ == exception;
        }

    protected:
         mutable mutex_type mtx_;
         state state_;                               // current state
    };

    template <typename Result>
    struct future_data : future_data<traits::detail::future_data_void>
    {
        HPX_NON_COPYABLE(future_data);

        typedef typename future_data_result<Result>::type result_type;
        typedef util::unique_function_nonser<void()> completed_callback_type;
        typedef lcos::local::spinlock mutex_type;
        typedef typename future_data<
                traits::detail::future_data_void
            >::init_no_addref init_no_addref;

        future_data()
        {}

        future_data(init_no_addref no_addref)
          : future_data<traits::detail::future_data_void>(no_addref)
        {}

        template <typename Target>
        future_data(Target && data, init_no_addref no_addref)
          : future_data<traits::detail::future_data_void>(no_addref)
        {
            result_type* value_ptr =
                reinterpret_cast<result_type*>(&storage_);
            ::new ((void*)value_ptr) result_type(
                future_data_result<Result>::set(std::forward<Target>(data)));
            state_ = value;
        }

        future_data(boost::exception_ptr const& e, init_no_addref no_addref)
          : future_data<traits::detail::future_data_void>(no_addref)
        {
            boost::exception_ptr* exception_ptr =
                reinterpret_cast<boost::exception_ptr*>(&storage_);
            ::new ((void*)exception_ptr) boost::exception_ptr(e);
            state_ = exception;
        }
        future_data(boost::exception_ptr && e, init_no_addref no_addref)
          : future_data<traits::detail::future_data_void>(no_addref)
        {
            boost::exception_ptr* exception_ptr =
                reinterpret_cast<boost::exception_ptr*>(&storage_);
            ::new ((void*)exception_ptr) boost::exception_ptr(std::move(e));
            state_ = exception;
        }

        virtual ~future_data() HPX_NOEXCEPT
        {
            reset();
        }

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
        virtual result_type* get_result(error_code& ec = throws)
        {
            // yields control if needed
            wait(ec);
            if (ec) return nullptr;

            // No locking is required. Once a future has been made ready, which
            // is a postcondition of wait, either:
            //
            // - there is only one writer (future), or
            // - there are multiple readers only (shared_future, lock hurts
            //   concurrency)

            if (state_ == empty) {
                // the value has already been moved out of this future
                HPX_THROWS_IF(ec, no_state,
                    "future_data::get_result",
                    "this future has no valid shared state");
                return nullptr;
            }

            // the thread has been re-activated by one of the actions
            // supported by this promise (see promise::set_event
            // and promise::set_exception).
            if (state_ == exception)
            {
                boost::exception_ptr* exception_ptr =
                    reinterpret_cast<boost::exception_ptr*>(&storage_);
                // an error has been reported in the meantime, throw or set
                // the error code
                if (&ec == &throws) {
                    boost::rethrow_exception(*exception_ptr);
                    // never reached
                }
                else {
                    ec = make_error_code(*exception_ptr);
                }
                return nullptr;
            }
            return reinterpret_cast<result_type*>(&storage_);
        }

        virtual util::unused_type* get_result_void(error_code& ec = throws)
        {
            // yields control if needed
            wait(ec);
            if (ec) return nullptr;

            // No locking is required. Once a future has been made ready, which
            // is a postcondition of wait, either:
            //
            // - there is only one writer (future), or
            // - there are multiple readers only (shared_future, lock hurts
            //   concurrency)

            if (state_ == empty) {
                // the value has already been moved out of this future
                HPX_THROWS_IF(ec, no_state,
                    "future_data::get_result",
                    "this future has no valid shared state");
                return nullptr;
            }

            // the thread has been re-activated by one of the actions
            // supported by this promise (see promise::set_event
            // and promise::set_exception).
            if (state_ == exception)
            {
                boost::exception_ptr* exception_ptr =
                    reinterpret_cast<boost::exception_ptr*>(&storage_);
                // an error has been reported in the meantime, throw or set
                // the error code
                if (&ec == &throws) {
                    boost::rethrow_exception(*exception_ptr);
                    // never reached
                }
                else {
                    ec = make_error_code(*exception_ptr);
                }
                return nullptr;
            }

            static util::unused_type unused_;
            return &unused_;
        }

        // deferred execution of a given continuation
        bool run_on_completed(completed_callback_type && on_completed,
            boost::exception_ptr& ptr)
        {
            try {
                hpx::util::annotate_function annotate(on_completed);
                (void)annotate;     // suppress warning about unused variable
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
            // We need to run the completion on a new thread if we are on a
            // non HPX thread.
            bool recurse_asynchronously = hpx::threads::get_self_ptr() == nullptr;
#if defined(HPX_HAVE_THREADS_GET_STACK_POINTER)
            recurse_asynchronously =
                !this_thread::has_sufficient_stack_space();
#else
            handle_continuation_recursion_count cnt;
            recurse_asynchronously = recurse_asynchronously ||
                cnt.count_ > HPX_CONTINUATION_MAX_RECURSION_DEPTH;
#endif
            if (!recurse_asynchronously)
            {
                // directly execute continuation on this thread
                boost::exception_ptr ptr;
                if (!run_on_completed(std::move(on_completed), ptr))
                {
                    error_code ec(lightweight);
                    set_exception(hpx::detail::access_exception(ec));
                }
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
                        set_exception(hpx::detail::access_exception(ec));
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
        void set_value(Target && data, error_code& ec = throws)
        {
            std::unique_lock<mutex_type> l(this->mtx_);

            // check whether the data has already been set
            if (is_ready_locked(l)) {
                l.unlock();
                HPX_THROWS_IF(ec, promise_already_satisfied,
                    "future_data::set_value",
                    "data has already been set for this future");
                return;
            }

            completed_callback_type on_completed = std::move(this->on_completed_);

            // set the data
            result_type* value_ptr =
                reinterpret_cast<result_type*>(&storage_);
            ::new ((void*)value_ptr) result_type(
                future_data_result<Result>::set(std::forward<Target>(data)));
            state_ = value;

            // handle all threads waiting for the future to become ready

            // Note: we use notify_one repeatedly instead of notify_all as we
            //       know: a) that most of the time we have at most one thread
            //       waiting on the future (most futures are not shared), and
            //       b) our implementation of condition_variable::notify_one
            //       relinquishes the lock before resuming the waiting thread
            //       which avoids suspension of this thread when it tries to
            //       re-lock the mutex while exiting from condition_variable::wait
            while (cond_.notify_one(std::move(l), threads::thread_priority_boost, ec))
            {
                l = std::unique_lock<mutex_type>(this->mtx_);
            }

            // Note: cv.notify_one() above 'consumes' the lock 'l' and leaves
            //       it unlocked when returning.

            // invoke the callback (continuation) function
            if (on_completed)
                handle_on_completed(std::move(on_completed));
        }

        template <typename Target>
        void set_exception(Target && data, error_code& ec = throws)
        {
            std::unique_lock<mutex_type> l(this->mtx_);

            // check whether the data has already been set
            if (is_ready_locked(l)) {
                l.unlock();
                HPX_THROWS_IF(ec, promise_already_satisfied,
                    "future_data::set_exception",
                    "data has already been set for this future");
                return;
            }

            completed_callback_type on_completed = std::move(this->on_completed_);

            // set the data
            boost::exception_ptr* exception_ptr =
                reinterpret_cast<boost::exception_ptr*>(&storage_);
            ::new ((void*)exception_ptr) boost::exception_ptr(
                std::forward<Target>(data));
            state_ = exception;

            // handle all threads waiting for the future to become ready

            // Note: we use notify_one repeatedly instead of notify_all as we
            //       know: a) that most of the time we have at most one thread
            //       waiting on the future (most futures are not shared), and
            //       b) our implementation of condition_variable::notify_one
            //       relinquishes the lock before resuming the waiting thread
            //       which avoids suspension of this thread when it tries to
            //       re-lock the mutex while exiting from condition_variable::wait
            while (cond_.notify_one(std::move(l), threads::thread_priority_boost, ec))
            {
                l = std::unique_lock<mutex_type>(this->mtx_);
            }

            // Note: cv.notify_one() above 'consumes' the lock 'l' and leaves
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
                set_value(std::move(get_remote_result_type::call(
                        std::forward<T>(result))));
            }
            catch (...) {
                // store the error instead
                return set_exception(boost::current_exception());
            }
        }

        // trigger the future with the given error condition
        void set_error(error e, char const* f, char const* msg)
        {
            try {
                HPX_THROW_EXCEPTION(e, f, msg);
            }
            catch (...) {
                // store the error code
                set_exception(boost::current_exception());
            }
        }

        /// Reset the promise to allow to restart an asynchronous
        /// operation. Allows any subsequent set_data operation to succeed.
        void reset(error_code& /*ec*/ = throws)
        {
            // no locking is required as semantics guarantee a single writer
            // and no reader

            // release any stored data and callback functions
            switch (state_) {
            case value:
            {
                result_type* value_ptr =
                    reinterpret_cast<result_type*>(&storage_);
                value_ptr->~result_type();
                break;
            }
            case exception:
            {
                boost::exception_ptr* exception_ptr =
                    reinterpret_cast<boost::exception_ptr*>(&storage_);
                exception_ptr->~exception_ptr();
                break;
            }
            default: break;
            }

            state_ = empty;
            on_completed_ = completed_callback_type();
        }

        // continuation support

        /// Set the callback which needs to be invoked when the future becomes
        /// ready. If the future is ready the function will be invoked
        /// immediately.
        void set_on_completed(completed_callback_type data_sink)
        {
            if (!data_sink) return;

            std::unique_lock<mutex_type> l(this->mtx_);

            if (is_ready_locked(l)) {

                HPX_ASSERT(!on_completed_);

                // invoke the callback (continuation) function right away
                l.unlock();

                handle_on_completed(std::move(data_sink));
            }
            else {
                // store a combined callback wrapping the old and the new one
                // make sure continuations are evaluated in the order they are
                // attached
                this->on_completed_ = compose_cb(
                    std::move(on_completed_), std::move(data_sink));
            }
        }

        virtual void wait(error_code& ec = throws)
        {
            std::unique_lock<mutex_type> l(mtx_);

            // block if this entry is empty
            if (state_ == empty) {
                cond_.wait(l, "future_data::wait", ec);
                if (ec) return;
            }

            if (&ec != &throws)
                ec = make_success_code();
        }

        virtual future_status
        wait_until(util::steady_clock::time_point const& abs_time,
            error_code& ec = throws)
        {
            std::unique_lock<mutex_type> l(mtx_);

            // block if this entry is empty
            if (state_ == empty) {
                threads::thread_state_ex_enum const reason =
                    cond_.wait_until(l, abs_time,
                        "future_data::wait_until", ec);
                if (ec) return future_status::uninitialized;

                if (reason == threads::wait_timeout)
                    return future_status::timeout;

                return future_status::ready;
            }

            if (&ec != &throws)
                ec = make_success_code();

            return future_status::ready; //-V110
        }

    protected:
        completed_callback_type on_completed_;

    private:
        local::detail::condition_variable cond_;    // threads waiting in read
        typename future_data_storage<Result>::type storage_;
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename Result, typename Allocator>
    struct future_data_allocator : future_data<Result>
    {
        typedef typename future_data<Result>::init_no_addref init_no_addref;
        typedef typename
                std::allocator_traits<Allocator>::template
                    rebind_alloc<future_data_allocator>
            other_allocator;

        future_data_allocator(other_allocator const& alloc)
          : future_data<Result>(), alloc_(alloc)
        {}
        future_data_allocator(init_no_addref no_addref,
                other_allocator const& alloc)
          : future_data<Result>(no_addref), alloc_(alloc)
        {}
        template <typename Target>
        future_data_allocator(Target && data, init_no_addref no_addref,
                other_allocator const& alloc)
          : future_data<Result>(std::move(data), no_addref), alloc_(alloc)
        {}
        future_data_allocator(boost::exception_ptr const& e,
                init_no_addref no_addref, other_allocator const& alloc)
          : future_data<Result>(e, no_addref), alloc_(alloc)
        {}
        future_data_allocator(boost::exception_ptr && e,
                init_no_addref no_addref, other_allocator const& alloc)
          : future_data<Result>(std::move(e), no_addref), alloc_(alloc)
        {}

    private:
        void destroy()
        {
            typedef std::allocator_traits<other_allocator> traits;

            other_allocator alloc(alloc_);
            traits::destroy(alloc, this);
            traits::deallocate(alloc, this, 1);
        }

    private:
        other_allocator alloc_;
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename Result>
    struct timed_future_data : future_data<Result>
    {
    public:
        typedef future_data<Result> base_type;
        typedef typename base_type::result_type result_type;
        typedef typename base_type::mutex_type mutex_type;

    public:
        timed_future_data() {}

        template <typename Result_>
        timed_future_data(
            util::steady_clock::time_point const& abs_time,
            Result_&& init)
        {
            boost::intrusive_ptr<timed_future_data> this_(this);

            error_code ec;
            threads::thread_id_type id = threads::register_thread_nullary(
                util::bind(util::one_shot(&timed_future_data::set_value),
                    std::move(this_),
                    future_data_result<Result>::set(std::forward<Result_>(init))),
                "timed_future_data<Result>::timed_future_data",
                threads::suspended, true, threads::thread_priority_boost,
                std::size_t(-1), threads::thread_stacksize_current, ec);
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

        void set_value(result_type const& value)
        {
            this->base_type::set_value(value);
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename Result>
    struct task_base : future_data<Result>
    {
    protected:
        typedef future_data<Result> base_type;
        typedef typename future_data<Result>::mutex_type mutex_type;
        typedef boost::intrusive_ptr<task_base> future_base_type;
        typedef typename future_data<Result>::result_type result_type;
        typedef typename base_type::init_no_addref init_no_addref;

    public:
        task_base()
          : started_(false), sched_(nullptr)
        {}

        task_base(init_no_addref no_addref)
          : base_type(no_addref), started_(false), sched_(nullptr)
        {}

        task_base(threads::executor& sched)
          : started_(false),
            sched_(sched ? &sched : nullptr)
        {}

        task_base(threads::executor& sched, init_no_addref no_addref)
          : base_type(no_addref),
            started_(false),
            sched_(sched ? &sched : nullptr)
        {}

        virtual void execute_deferred(error_code& ec = throws)
        {
            if (!started_test_and_set())
                this->do_run();
        }

        // retrieving the value
        virtual result_type* get_result(error_code& ec = throws)
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
            this->future_data<Result>::wait(ec);
        }

        virtual future_status
        wait_until(util::steady_clock::time_point const& abs_time,
            error_code& ec = throws)
        {
            if (!started_test())
                return future_status::deferred; //-V110
            return this->future_data<Result>::wait_until(abs_time, ec);
        }

    private:
        bool started_test() const
        {
            std::lock_guard<mutex_type> l(this->mtx_);
            return started_;
        }

        bool started_test_and_set()
        {
            std::lock_guard<mutex_type> l(this->mtx_);
            return started_test_and_set_locked(l);
        }

        template <typename Lock>
        bool started_test_and_set_locked(Lock& l)
        {
            HPX_ASSERT_OWNS_LOCK(l);
            if (started_)
                return true;

            started_ = true;
            return false;
        }

    protected:
        void check_started()
        {
            std::unique_lock<mutex_type> l(this->mtx_);
            if (started_) {
                l.unlock();
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
        virtual threads::thread_id_type apply(launch policy,
            threads::thread_priority priority,
            threads::thread_stacksize stacksize, error_code& ec)
        {
            HPX_ASSERT(false);      // shouldn't ever be called
            return threads::invalid_thread_id;
        }

    protected:
        static threads::thread_result_type run_impl(future_base_type this_)
        {
            this_->do_run();
            return threads::thread_result_type(threads::terminated, nullptr);
        }

    public:
        template <typename T>
        void set_data(T && result)
        {
            this->future_data<Result>::set_data(std::forward<T>(result));
        }

        void set_exception(
            boost::exception_ptr const& e, error_code& ec = throws)
        {
            this->future_data<Result>::set_exception(e, ec);
        }

        virtual void do_run()
        {
            HPX_ASSERT(false);      // shouldn't ever be called
        }

    protected:
        bool started_;
        threads::executor* sched_;
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename Result>
    struct cancelable_task_base : task_base<Result>
    {
    protected:
        typedef typename task_base<Result>::mutex_type mutex_type;
        typedef boost::intrusive_ptr<cancelable_task_base> future_base_type;
        typedef typename future_data<Result>::result_type result_type;
        typedef typename task_base<Result>::init_no_addref init_no_addref;

    protected:
        threads::thread_id_type get_thread_id() const
        {
            std::lock_guard<mutex_type> l(this->mtx_);
            return id_;
        }
        void set_thread_id(threads::thread_id_type id)
        {
            std::lock_guard<mutex_type> l(this->mtx_);
            id_ = id;
        }

    public:
        cancelable_task_base()
          : id_(threads::invalid_thread_id)
        {}

        cancelable_task_base(init_no_addref no_addref)
          : task_base<Result>(no_addref), id_(threads::invalid_thread_id)
        {}

        cancelable_task_base(threads::executor& sched)
          : task_base<Result>(sched), id_(threads::invalid_thread_id)
        {}

        cancelable_task_base(threads::executor& sched, init_no_addref no_addref)
          : task_base<Result>(sched, no_addref), id_(threads::invalid_thread_id)
        {}

    private:
        struct reset_id
        {
            reset_id(cancelable_task_base& target)
              : target_(target)
            {
                target.set_thread_id(threads::get_self_id());
            }
            ~reset_id()
            {
                target_.set_thread_id(threads::invalid_thread_id);
            }
            cancelable_task_base& target_;
        };

    protected:
        static threads::thread_result_type run_impl(future_base_type this_)
        {
            reset_id r(*this_);
            this_->do_run();
            return threads::thread_result_type(threads::terminated, nullptr);
        }

    public:
        // cancellation support
        bool cancelable() const
        {
            return true;
        }

        void cancel()
        {
            std::unique_lock<mutex_type> l(this->mtx_);
            try {
                if (!this->started_)
                    HPX_THROW_THREAD_INTERRUPTED_EXCEPTION();

                if (this->is_ready_locked(l))
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
                    l.unlock();
                    HPX_THROW_EXCEPTION(future_can_not_be_cancelled,
                        "task_base<Result>::cancel",
                        "future can't be canceled at this time");
                }
            }
            catch (...) {
                this->started_ = true;
                this->set_exception(boost::current_exception());
                throw;
            }
        }

    protected:
        threads::thread_id_type id_;
    };
}}}

namespace hpx { namespace traits { namespace detail
{
    template <typename R, typename Allocator>
    struct shared_state_allocator<lcos::detail::future_data<R>, Allocator>
    {
        typedef lcos::detail::future_data_allocator<R, Allocator> type;
    };
}}}

#endif
