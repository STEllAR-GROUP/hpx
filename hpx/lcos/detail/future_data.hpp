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
#include <hpx/runtime/launch_policy.hpp>
#include <hpx/runtime/threads/coroutines/detail/get_stack_pointer.hpp>
#include <hpx/runtime/threads/thread_executor.hpp>
#include <hpx/runtime/threads/thread_helpers.hpp>
#include <hpx/throw_exception.hpp>
#include <hpx/traits/future_access.hpp>
#include <hpx/traits/get_remote_result.hpp>
#include <hpx/util/annotated_function.hpp>
#include <hpx/util/assert.hpp>
#include <hpx/util/assert_owns_lock.hpp>
#include <hpx/util/atomic_count.hpp>
#include <hpx/util/bind.hpp>
#include <hpx/util/decay.hpp>
#include <hpx/util/steady_clock.hpp>
#include <hpx/util/unique_function.hpp>
#include <hpx/util/unused.hpp>

#include <boost/intrusive_ptr.hpp>

#if BOOST_VERSION < 105800
#include <vector>
#else
#include <boost/container/small_vector.hpp>
#endif

#include <chrono>
#include <cstddef>
#include <exception>
#include <memory>
#include <mutex>
#include <string>
#include <type_traits>
#include <utility>

#include <hpx/config/warnings_prefix.hpp>

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
    struct HPX_EXPORT future_data_refcnt_base
    {
    public:
        typedef util::unique_function_nonser<void()> completed_callback_type;
#if BOOST_VERSION < 105800
        typedef std::vector<completed_callback_type>
            completed_callback_vector_type;
#else
        typedef boost::container::small_vector<completed_callback_type, 3>
            completed_callback_vector_type;
#endif

        typedef void has_future_data_refcnt_base;

        virtual ~future_data_refcnt_base();

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
        typedef std::exception_ptr error_type;

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
    template <typename Result>
    struct future_data_base;

    template <>
    struct HPX_EXPORT future_data_base<traits::detail::future_data_void>
      : future_data_refcnt_base
    {
        future_data_base()
          : state_(empty)
        {}

        future_data_base(init_no_addref no_addref)
          : future_data_refcnt_base(no_addref), state_(empty)
        {}

        using future_data_refcnt_base::completed_callback_type;
        using future_data_refcnt_base::completed_callback_vector_type;
        typedef lcos::local::spinlock mutex_type;
        typedef util::unused_type result_type;
        typedef future_data_refcnt_base::init_no_addref init_no_addref;

        virtual ~future_data_base();

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

        virtual void execute_deferred(error_code& /*ec*/ = throws) {}

        // cancellation is disabled by default
        virtual bool cancelable() const
        {
            return false;
        }
        virtual void cancel()
        {
            HPX_THROW_EXCEPTION(future_does_not_support_cancellation,
                "future_data_base::cancel",
                "this future does not support cancellation");
        }

        result_type* get_result_void(void const* storage, error_code& ec = throws);
        virtual result_type* get_result_void(error_code& ec = throws) = 0;

        virtual void set_exception(std::exception_ptr data,
            error_code& ec = throws) = 0;

        // continuation support

        // deferred execution of a given continuation
        bool run_on_completed(completed_callback_vector_type && on_completed,
            std::exception_ptr& ptr);

        // make sure continuation invocation does not recurse deeper than
        // allowed
        void handle_on_completed(completed_callback_vector_type && on_completed);

        /// Set the callback which needs to be invoked when the future becomes
        /// ready. If the future is ready the function will be invoked
        /// immediately.
        void set_on_completed(completed_callback_type data_sink) override;

        virtual void wait(error_code& ec = throws);

        virtual future_status wait_until(
            util::steady_clock::time_point const& abs_time, error_code& ec = throws);

        virtual std::exception_ptr get_exception_ptr() const = 0;

        virtual std::string const& get_registered_name() const
        {
            HPX_THROW_EXCEPTION(invalid_status,
                "future_data_base::get_registered_name",
                "this future does not support name registration");
        }
        virtual void register_as(std::string const& name, bool manage_lifetime)
        {
            HPX_THROW_EXCEPTION(invalid_status,
                "future_data_base::set_registered_name",
                "this future does not support name registration");
        }

    protected:
        mutable mutex_type mtx_;
        state state_;                               // current state
        completed_callback_vector_type on_completed_;
        local::detail::condition_variable cond_;    // threads waiting in read
    };

    template <typename Result>
    struct future_data_base
      : future_data_base<traits::detail::future_data_void>
    {
        HPX_NON_COPYABLE(future_data_base);

        typedef typename future_data_result<Result>::type result_type;
        typedef future_data_base<traits::detail::future_data_void> base_type;
        typedef lcos::local::spinlock mutex_type;
        typedef typename base_type::init_no_addref init_no_addref;
        typedef typename base_type::completed_callback_type completed_callback_type;
        typedef typename base_type::completed_callback_vector_type
            completed_callback_vector_type;

        future_data_base() = default;

        future_data_base(init_no_addref no_addref)
          : base_type(no_addref)
        {}

        template <typename Target>
        future_data_base(Target && data, init_no_addref no_addref)
          : base_type(no_addref)
        {
            result_type* value_ptr =
                reinterpret_cast<result_type*>(&storage_);
            ::new ((void*)value_ptr) result_type(
                future_data_result<Result>::set(std::forward<Target>(data)));
            state_ = value;
        }

        future_data_base(std::exception_ptr const& e, init_no_addref no_addref)
          : base_type(no_addref)
        {
            std::exception_ptr* exception_ptr =
                reinterpret_cast<std::exception_ptr*>(&storage_);
            ::new ((void*)exception_ptr) std::exception_ptr(e);
            state_ = exception;
        }
        future_data_base(std::exception_ptr && e, init_no_addref no_addref)
          : base_type(no_addref)
        {
            std::exception_ptr* exception_ptr =
                reinterpret_cast<std::exception_ptr*>(&storage_);
            ::new ((void*)exception_ptr) std::exception_ptr(std::move(e));
            state_ = exception;
        }

        virtual ~future_data_base() noexcept
        {
            reset();
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
            if (!get_result_void(ec))
                return nullptr;
            return reinterpret_cast<result_type*>(&storage_);
        }

        util::unused_type* get_result_void(error_code& ec = throws) override
        {
            return base_type::get_result_void(&storage_, ec);
        }

        /// Set the result of the requested action.
        template <typename Target>
        void set_value(Target && data, error_code& ec = throws)
        {
            std::unique_lock<mutex_type> l(mtx_);

            // check whether the data has already been set
            if (is_ready_locked(l)) {
                l.unlock();
                HPX_THROWS_IF(ec, promise_already_satisfied,
                    "future_data_base::set_value",
                    "data has already been set for this future");
                return;
            }

            auto on_completed = std::move(on_completed_);
            on_completed_ = completed_callback_vector_type();

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
                l = std::unique_lock<mutex_type>(mtx_);
            }

            // Note: cv.notify_one() above 'consumes' the lock 'l' and leaves
            //       it unlocked when returning.

            // invoke the callback (continuation) function
            if (!on_completed.empty())
                handle_on_completed(std::move(on_completed));
        }

        void set_exception(
            std::exception_ptr data, error_code& ec = throws) override
        {
            std::unique_lock<mutex_type> l(mtx_);

            // check whether the data has already been set
            if (is_ready_locked(l)) {
                l.unlock();
                HPX_THROWS_IF(ec, promise_already_satisfied,
                    "future_data_base::set_exception",
                    "data has already been set for this future");
                return;
            }

            auto on_completed = std::move(on_completed_);
            on_completed_ = completed_callback_vector_type();

            // set the data
            std::exception_ptr* exception_ptr =
                reinterpret_cast<std::exception_ptr*>(&storage_);
            ::new ((void*)exception_ptr) std::exception_ptr(std::move(data));
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
                l = std::unique_lock<mutex_type>(mtx_);
            }

            // Note: cv.notify_one() above 'consumes' the lock 'l' and leaves
            //       it unlocked when returning.

            // invoke the callback (continuation) function
            if (!on_completed.empty())
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
                return set_exception(std::current_exception());
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
                set_exception(std::current_exception());
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
                std::exception_ptr* exception_ptr =
                    reinterpret_cast<std::exception_ptr*>(&storage_);
                exception_ptr->~exception_ptr();
                break;
            }
            default: break;
            }

            state_ = empty;
            on_completed_ = completed_callback_vector_type();
        }

        std::exception_ptr get_exception_ptr() const override
        {
            HPX_ASSERT(state_ == exception);
            return *reinterpret_cast<std::exception_ptr const*>(&storage_);
        }

    protected:
        using base_type::mtx_;
        using base_type::state_;
        using base_type::on_completed_;

    private:
        using base_type::cond_;
        typename future_data_storage<Result>::type storage_;
    };

    ///////////////////////////////////////////////////////////////////////////
    // Customization point to have the ability for creating distinct shared
    // states depending on the value type held.
    template <typename Result>
    struct future_data : future_data_base<Result>
    {
        HPX_NON_COPYABLE(future_data);

        typedef typename future_data_base<Result>::init_no_addref init_no_addref;

        future_data() = default;

        future_data(init_no_addref no_addref)
          : future_data_base<Result>(no_addref)
        {}

        template <typename Target>
        future_data(Target && data, init_no_addref no_addref)
          : future_data_base<Result>(std::forward<Target>(data), no_addref)
        {}

        future_data(std::exception_ptr const& e, init_no_addref no_addref)
          : future_data_base<Result>(e, no_addref)
        {}
        future_data(std::exception_ptr && e, init_no_addref no_addref)
          : future_data_base<Result>(std::move(e), no_addref)
        {}

        ~future_data() noexcept override = default;
    };

    // Specialization for shared state of id_type, additionally (optionally)
    // holds a registered name for the object it refers to.
    template <> struct future_data<id_type>;

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
          : future_data<Result>(std::forward<Target>(data), no_addref), alloc_(alloc)
        {}
        future_data_allocator(std::exception_ptr const& e,
                init_no_addref no_addref, other_allocator const& alloc)
          : future_data<Result>(e, no_addref), alloc_(alloc)
        {}
        future_data_allocator(std::exception_ptr && e,
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
          : started_(false)
        {}

        task_base(init_no_addref no_addref)
          : base_type(no_addref), started_(false)
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
        bool started_test_and_set()
        {
            std::lock_guard<mutex_type> l(this->mtx_);
            return started_test_and_set_locked(l);
        }

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
            return threads::thread_result_type(
                threads::terminated, threads::invalid_thread_id);
        }

    public:
        template <typename T>
        void set_data(T && result)
        {
            this->future_data<Result>::set_data(std::forward<T>(result));
        }

        void set_exception(
            std::exception_ptr e, error_code& ec = throws)
        {
            this->future_data<Result>::set_exception(std::move(e), ec);
        }

        virtual void do_run()
        {
            HPX_ASSERT(false);      // shouldn't ever be called
        }

    protected:
        bool started_;
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
            return threads::thread_result_type(
                threads::terminated, threads::invalid_thread_id);
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
                this->set_exception(std::current_exception());
                throw;
            }
        }

    protected:
        threads::thread_id_type id_;
    };
}}}

namespace hpx { namespace traits
{
    namespace detail
    {
        template <typename R, typename Allocator>
        struct shared_state_allocator<lcos::detail::future_data<R>, Allocator>
        {
            typedef lcos::detail::future_data_allocator<R, Allocator> type;
        };
    }
}}

#include <hpx/config/warnings_suffix.hpp>

#endif
