//  Copyright (c) 2007-2013 Hartmut Kaiser
//  Copyright (c) 2013 Agustin Berge
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_LCOS_FUTURE_MAR_06_2012_1059AM)
#define HPX_LCOS_FUTURE_MAR_06_2012_1059AM

#include <hpx/hpx_fwd.hpp>
#include <hpx/config/forceinline.hpp>
#include <hpx/traits/is_future.hpp>
#include <hpx/lcos/detail/future_data.hpp>
#include <hpx/util/always_void.hpp>
#include <hpx/util/date_time_chrono.hpp>
#include <hpx/util/decay.hpp>
#include <hpx/util/move.hpp>
#include <hpx/util/result_of.hpp>
#include <hpx/runtime/actions/continuation.hpp>

#include <boost/intrusive_ptr.hpp>
#include <boost/date_time/posix_time/ptime.hpp>
#include <boost/function_types/result_type.hpp>
#include <boost/detail/iterator.hpp>
#include <boost/static_assert.hpp>
#include <boost/type_traits/is_void.hpp>
#include <boost/mpl/eval_if.hpp>
#include <boost/mpl/if.hpp>
#include <boost/utility/declval.hpp>

namespace hpx { namespace lcos { namespace detail
{
    ///////////////////////////////////////////////////////////////////////////
    template <typename T>
    struct future_traits
    {};

    template <typename Future>
    struct future_traits<Future const>
      : future_traits<Future>
    {};

    template <typename Future>
    struct future_traits<Future&>
      : future_traits<Future>
    {};

    template <typename Future>
    struct future_traits<Future const &>
      : future_traits<Future>
    {};

#if defined(HPX_ENABLE_DEPRECATED_FUTURE)
    template <typename T>
    struct future_traits<lcos::future<T> >
    {
        typedef T type;
    };
#endif

    template <typename R>
    struct future_traits<lcos::unique_future<R> >
    {
        typedef R type;
    };

    template <typename R>
    struct future_traits<lcos::shared_future<R> >
    {
        typedef R type;
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename R>
    struct shared_state_ptr
    {
        typedef boost::intrusive_ptr<future_data<R> > type;
    };

    template <typename Future>
    struct shared_state_ptr_for
    {};
    
    template <typename Future>
    struct shared_state_ptr_for<Future const>
      : shared_state_ptr_for<Future>
    {};
    
    template <typename Future>
    struct shared_state_ptr_for<Future&>
      : shared_state_ptr_for<Future>
    {};
    
    template <typename Future>
    struct shared_state_ptr_for<Future &&>
      : shared_state_ptr_for<Future>
    {};

    template <typename R>
    struct shared_state_ptr_for<unique_future<R> >
      : shared_state_ptr<R>
    {};

    template <typename R>
    struct shared_state_ptr_for<shared_future<R> >
      : shared_state_ptr<R>
    {};

#if defined(HPX_ENABLE_DEPRECATED_FUTURE)
    template <typename R>
    struct shared_state_ptr_for<future<R> >
      : shared_state_ptr<R>
    {};
#endif

    ///////////////////////////////////////////////////////////////////////////
    struct future_access
    {
        template <typename Future, typename SharedState>
        static Future
        create(boost::intrusive_ptr<SharedState> const& shared_state)
        {
            return Future(shared_state);
        }

        template <typename Future, typename SharedState>
        static Future
        create(boost::intrusive_ptr<SharedState> && shared_state)
        {
            return Future(std::move(shared_state));
        }

        template <typename Future, typename SharedState>
        static Future
        create(SharedState* shared_state)
        {
            return Future(boost::intrusive_ptr<SharedState>(shared_state));
        }

        template <typename R>
        BOOST_FORCEINLINE static
        typename shared_state_ptr<R>::type const&
        get_shared_state(unique_future<R> const& f)
        {
            return f.shared_state_;
        }

        template <typename R>
        BOOST_FORCEINLINE static
        typename shared_state_ptr<R>::type const&
        get_shared_state(shared_future<R> const& f)
        {
            return f.shared_state_;
        }

#if defined(HPX_ENABLE_DEPRECATED_FUTURE)
        template <typename R>
        BOOST_FORCEINLINE static
        typename shared_state_ptr<R>::type const&
        get_shared_state(future<R> const& f)
        {
            return f.future_data_;
        }
#endif
    };

#if defined(HPX_ENABLE_DEPRECATED_FUTURE)
    ///////////////////////////////////////////////////////////////////////////
    template <typename Result>
    inline lcos::future<Result> make_future_from_data(
        boost::intrusive_ptr<detail::future_data<Result> > const& p)
    {
        return future_access::create<lcos::future<Result> >(p);
    }

    template <typename Result>
    inline lcos::future<Result> make_future_from_data( //-V659
        boost::intrusive_ptr<detail::future_data<Result> > && p)
    {
        return future_access::create<lcos::future<Result> >(std::move(p));
    }

    template <typename Result>
    inline lcos::future<Result> make_future_from_data(
        detail::future_data<Result>* p)
    {
        boost::intrusive_ptr<detail::future_data<Result> > shared_state_ptr(p);
        return future_access::create<lcos::future<Result> >(std::move(shared_state_ptr));
    }

    template <typename Result>
    inline detail::future_data<Result>*
        get_future_data(lcos::future<Result> const& f)
    {
        return future_access::get_shared_state(f).get();
    }
#endif

    ///////////////////////////////////////////////////////////////////////////
    template <typename Future, typename F, typename Enable = void>
    struct future_then_result
    {
        typedef struct continuation_not_callable
        {
            void error(Future future, F& f)
            {
                f(future);
            }

            ~continuation_not_callable()
            {
                error(boost::declval<Future>(), boost::declval<F&>());
            }
        } type;
    };

    template <typename Future, typename F>
    struct future_then_result<
        Future, F
      , typename util::always_void<
            typename util::result_of<F(Future)>::type
        >::type
    >
    {
        typedef typename util::result_of<F(Future)>::type result;

        typedef lcos::unique_future<
            typename boost::mpl::eval_if<
                traits::is_future<result>
              , future_traits<result>
              , boost::mpl::identity<result>
            >::type> type;
    };

#if defined(HPX_ENABLE_DEPRECATED_FUTURE)
    template <typename Result, typename F>
    struct future_then_result<
        future<Result>, F
      , typename util::always_void<
            typename util::result_of<F(future<Result>)>::type
        >::type
    >
    {
        typedef typename util::result_of<F(future<Result>)>::type result;

        typedef
            typename boost::mpl::if_<
                traits::is_future<result>
              , result
              , lcos::future<result>
            >::type
            type;
    };
#endif

    ///////////////////////////////////////////////////////////////////////////
    template <typename Future, typename Enable = void>
    struct future_unwrap_result
    {
        typedef typename future_traits<Future>::type outer_result;

        typedef lcos::unique_future<
            typename boost::mpl::eval_if<
                traits::is_future<outer_result>
              , future_traits<outer_result>
              , boost::mpl::identity<void>
            >::type> type;
    };

#if defined(HPX_ENABLE_DEPRECATED_FUTURE)
    template <typename Result>
    struct future_unwrap_result<future<Result> >
      : boost::mpl::if_<traits::is_future<Result>, Result, void>
    {};
#endif

    ///////////////////////////////////////////////////////////////////////////
    template <typename Iter>
    struct future_iterator_traits
    {
        typedef
            typename boost::detail::iterator_traits<Iter>::value_type
            type;

        typedef future_traits<type> traits_type;
    };

#if defined(HPX_ENABLE_DEPRECATED_FUTURE)
    template <typename T>
    struct future_iterator_traits<future<T> >
    {};
#endif

    template <typename T>
    struct future_iterator_traits<unique_future<T> >
    {};

    template <typename T>
    struct future_iterator_traits<shared_future<T> >
    {};

    ///////////////////////////////////////////////////////////////////////////
    template <typename T>
    struct future_value
    {
        typedef T type;
        typedef T const& const_lvref;

        template <typename U>
        BOOST_FORCEINLINE static
        U get(U && u)
        {
            return std::forward<U>(u);
        }

        static T get_default()
        {
            return T();
        }
    };

    template <typename T>
    struct future_value<T&>
    {
        typedef T* type;
        typedef T& const_lvref;

        BOOST_FORCEINLINE static
        T& get(T* u)
        {
            return *u;
        }

        static T& get_default()
        {
            static T default_;
            return default_;
        }
    };

    template <>
    struct future_value<void>
    {
        typedef void type;
        typedef void const_lvref;

        BOOST_FORCEINLINE static
        void get(util::unused_type)
        {}

        static void get_default()
        {}
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename Future, typename F, typename ContResult>
    class continuation;

    template <typename ContResult, typename Future, typename F>
    inline typename shared_state_ptr<ContResult>::type
    make_continuation(Future& future, BOOST_SCOPED_ENUM(launch) policy,
        F && f);

    template <typename ContResult, typename Future, typename F>
    inline typename shared_state_ptr<ContResult>::type
    make_continuation(Future& future, threads::executor& sched,
        F && f);

    ///////////////////////////////////////////////////////////////////////////
    template <typename Future>
    struct unwrap_result
    {
        typedef typename future_traits<Future>::type outer_result;
        typedef
            typename boost::mpl::eval_if<
                traits::is_future<outer_result>
              , future_traits<outer_result>
              , boost::mpl::identity<void>
            >::type
            type;
    };

    template <typename Future>
    typename shared_state_ptr<
        typename unwrap_result<Future>::type>::type
    unwrap(Future& future, error_code& ec);

    ///////////////////////////////////////////////////////////////////////////
    template <typename Derived, typename R>
    class future_base
    {
    public:
        typedef R result_type;
        typedef future_data<R> shared_state_type;

    public:
        future_base() BOOST_NOEXCEPT
          : shared_state_()
        {}

        explicit future_base(
            boost::intrusive_ptr<shared_state_type> const& p
        ) : shared_state_(p)
        {}

        explicit future_base(
            boost::intrusive_ptr<shared_state_type> && p
        ) : shared_state_(std::move(p))
        {}

        future_base(future_base const& other)
          : shared_state_(other.shared_state_)
        {}

        future_base(future_base && other) BOOST_NOEXCEPT
          : shared_state_(std::move(other.shared_state_))
        {
            other.shared_state_ = 0;
        }

        void swap(future_base& other)
        {
            shared_state_.swap(other.shared_state_);
        }

        future_base& operator=(future_base const & other)
        {
            if (this != &other)
            {
                shared_state_ = other.shared_state_;
            }
            return *this;
        }

        future_base& operator=(future_base && other) BOOST_NOEXCEPT
        {
            if (this != &other)
            {
                shared_state_ = std::move(other.shared_state_);
                other.shared_state_ = 0;
            }
            return *this;
        }

        // Returns: true only if *this refers to a shared state.
        bool valid() const BOOST_NOEXCEPT
        {
            return shared_state_ != 0;
        }

        // Returns: true if the shared state is ready, false if it isn't.
        bool is_ready() const
        {
            return shared_state_ != 0 && shared_state_->is_ready();
        }

        // Returns: true if the shared state is ready and stores a value,
        //          false if it isn't.
        bool has_value() const
        {
            return shared_state_ != 0 && shared_state_->has_value();
        }

        // Returns: true if the shared state is ready and stores an exception,
        //          false if it isn't.
        bool has_exception() const
        {
            return shared_state_ != 0 && shared_state_->has_exception();
        }

        // Returns the future status
        BOOST_SCOPED_ENUM(future_status) get_status() const
        {
            if (!shared_state_)
                return future_status::uninitialized;

            return shared_state_->get_status();
        }

        // Notes: The three functions differ only by input parameters.
        //   - The first only takes a callable object which accepts a future
        //     object as a parameter.
        //   - The second function takes an executor as the first parameter
        //     and a callable object as the second parameter.
        //   - The third function takes a launch policy as the first parameter
        //     and a callable object as the second parameter.
        //   In cases where 'decltype(func(*this))' is future<R>, the
        //   resulting type is future<R> instead of future<future<R>>.
        // Effects:
        //   - The continuation is called when the object's shared state is
        //     ready (has a value or exception stored).
        //   - The continuation launches according to the specified launch
        //     policy or executor.
        //   - When the executor or launch policy is not provided the
        //     continuation inherits the parent's launch policy or executor.
        //   - If the parent was created with std::promise or with a
        //     packaged_task (has no associated launch policy), the
        //     continuation behaves the same as the third overload with a
        //     policy argument of launch::async | launch::deferred and the
        //     same argument for func.
        //   - If the parent has a policy of launch::deferred and the
        //     continuation does not have a specified launch policy or
        //     scheduler, then the parent is filled by immediately calling
        //     .wait(), and the policy of the antecedent is launch::deferred
        // Returns: An object of type future<decltype(func(*this))> that
        //          refers to the shared state created by the continuation.
        // Postcondition:
        //   - The future object is moved to the parameter of the continuation
        //     function.
        //   - valid() == false on original future object immediately after it
        //     returns.
        template <typename F>
        typename future_then_result<Derived, F>::type
        then(F && f)
        {
            return then(launch::all, std::forward<F>(f));
        }

        template <typename F>
        typename future_then_result<Derived, F>::type
        then(BOOST_SCOPED_ENUM(launch) policy, F && f)
        {
            typedef
                typename future_then_result<Derived, F>::type
                result_type;

            if (!shared_state_)
            {
                HPX_THROW_EXCEPTION(no_state,
                    "future_base<R>::then",
                    "this future has no valid shared state");
                return result_type();
            }

            typedef typename util::result_of<F(Derived)>::type result;
            typedef typename shared_state_ptr<result>::type shared_state_ptr;

            shared_state_ptr p =
                detail::make_continuation<result>(*static_cast<Derived*>(this),
                    policy, std::forward<F>(f));
            return future_access::create<unique_future<result> >(std::move(p));
        }

        template <typename F>
        typename future_then_result<Derived, F>::type
        then(threads::executor& sched, F && f)
        {
            typedef
                typename future_then_result<Derived, F>::type
                result_type;

            if (!shared_state_)
            {
                HPX_THROW_EXCEPTION(no_state,
                    "future_base<R>::then",
                    "this future has no valid shared state");
                return result_type();
            }

            typedef typename util::result_of<F(Derived)>::type result;
            typedef typename shared_state_ptr<result>::type shared_state_ptr;

            shared_state_ptr p =
                detail::make_continuation<result>(*static_cast<Derived*>(this),
                    sched, std::forward<F>(f));
            return future_access::create<unique_future<result> >(std::move(p));
        }

        // Notes:
        //   - R is a future<R2> or shared_future<R2>
        //   - Removes the outer-most future and returns a proxy to the inner
        //     future. The proxy is a representation of the inner future and
        //     it holds the same value (or exception) as the inner future.
        // Effects:
        //   - future<R2> X = future<future<R2>>.unwrap(), returns a future<R2>
        //     that becomes ready when the shared state of the inner future is
        //     ready. When the inner future is ready, its value (or exception)
        //     is moved to the shared state of the returned future.
        //   - future<R2> Y = future<shared_future<R2>>.unwrap(),returns a
        //     future<R2> that becomes ready when the shared state of the inner
        //     future is ready. When the inner shared_future is ready, its
        //     value (or exception) is copied to the shared state of the
        //     returned future.
        //   - If the outer future throws an exception, and .get() is called on
        //     the returned future, the returned future throws the same
        //     exception as the outer future. This is the case because the
        //     inner future didn't exit.
        // Returns: a future of type R2. The result of the inner future is
        //          moved out (shared_future is copied out) and stored in the
        //          shared state of the returned future when it is ready or the
        //          result of the inner future throws an exception.
        // Postcondition:
        //   - The returned future has valid() == true, regardless of the
        //     validity of the inner future.
        typename future_unwrap_result<Derived>::type
        unwrap(error_code& ec = throws)
        {
            BOOST_STATIC_ASSERT_MSG(
                traits::is_future<R>::value, "invalid use of unwrap");

            typedef
                typename future_unwrap_result<Derived>::type
                result_type;

            if (!shared_state_) {
                HPX_THROWS_IF(ec, no_state,
                    "future_base<R>::unwrap",
                    "this future has not been initialized");
                return result_type();
            }

            typedef
                typename shared_state_ptr_for<result_type>::type
                shared_state_ptr;

            shared_state_ptr state =
                lcos::detail::unwrap(*static_cast<Derived*>(this), ec);
            return future_access::create<result_type>(std::move(state));
        }

        // Effects: blocks until the shared state is ready.
        void wait(error_code& ec = throws) const
        {
            if (!shared_state_)
            {
                HPX_THROWS_IF(ec, no_state,
                    "future_base<R>::wait",
                    "this future has no valid shared state");
                return;
            }
            shared_state_->wait(ec);
        }

        // Effects: none if the shared state contains a deferred function
        //          (30.6.8), otherwise blocks until the shared state is ready
        //          or until the relative timeout (30.2.4) specified by
        //          rel_time has expired.
        // Returns:
        //   - future_status::deferred if the shared state contains a deferred
        //     function.
        //   - future_status::ready if the shared state is ready.
        //   - future_status::timeout if the function is returning because the
        //     relative timeout (30.2.4) specified by rel_time has expired.
        // Throws: timeout-related exceptions (30.2.4).
        BOOST_SCOPED_ENUM(future_status)
        wait_for(boost::posix_time::time_duration const& rel_time,
            error_code& ec = throws) const
        {
            if (!shared_state_)
            {
                HPX_THROWS_IF(ec, no_state,
                    "future_base<R>::wait_for",
                    "this future has no valid shared state");
                return future_status::uninitialized;
            }
            return shared_state_->wait_for(rel_time, ec);
        }
        template <class Rep, class Period>
        BOOST_SCOPED_ENUM(future_status)
        wait_for(boost::chrono::duration<Rep, Period> const& rel_time,
            error_code& ec = throws) const
        {
            return wait_for(util::to_time_duration(rel_time), ec);
        }

        // Effects: none if the shared state contains a deferred function
        //          (30.6.8), otherwise blocks until the shared state is ready
        //          or until the absolute timeout (30.2.4) specified by
        //          abs_time has expired.
        // Returns:
        //   - future_status::deferred if the shared state contains a deferred
        //     function.
        //   - future_status::ready if the shared state is ready.
        //   - future_status::timeout if the function is returning because the
        //     absolute timeout (30.2.4) specified by abs_time has expired.
        // Throws: timeout-related exceptions (30.2.4).
        BOOST_SCOPED_ENUM(future_status)
        wait_until(boost::posix_time::ptime const& abs_time,
            error_code& ec = throws) const
        {
            if (!shared_state_)
            {
                HPX_THROWS_IF(ec, no_state,
                    "future_base<R>::wait_until",
                    "this future has no valid shared state");
                return future_status::uninitialized;
            }
            return shared_state_->wait_until(abs_time, ec);
        }
        template <class Clock, class Duration>
        BOOST_SCOPED_ENUM(future_status)
        wait_until(boost::chrono::time_point<Clock, Duration> const& abs_time,
            error_code& ec = throws) const
        {
            return wait_until(util::to_ptime(abs_time), ec);
        }

    protected:
        boost::intrusive_ptr<shared_state_type> shared_state_;
    };
}}}

namespace hpx { namespace lcos
{
    ///////////////////////////////////////////////////////////////////////////
    // [N3722, 4.1] asks for this...
    namespace local
    {
        template <typename Result>
        class promise;
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename R>
    class unique_future : public detail::future_base<unique_future<R>, R>
    {
        HPX_MOVABLE_BUT_NOT_COPYABLE(unique_future);

        typedef detail::future_base<unique_future<R>, R> base_type;

    public:
        typedef R result_type;
        typedef typename base_type::shared_state_type shared_state_type;

    private:
        struct invalidate
        {
            explicit invalidate(unique_future& f)
              : f_(f)
            {}

            ~invalidate()
            {
                f_.shared_state_ = 0;
            }

            unique_future& f_;
        };

    private:
        friend struct detail::future_access;

        // Effects: constructs a future object from an shared state
        explicit unique_future(
            boost::intrusive_ptr<shared_state_type> const& state
        ) : base_type(state)
        {}

        explicit unique_future(
            boost::intrusive_ptr<shared_state_type> && state
        ) : base_type(std::move(state))
        {}

        template <typename SharedState>
        explicit unique_future(boost::intrusive_ptr<SharedState> const& state)
          : base_type(boost::static_pointer_cast<shared_state_type>(state))
        {}

    public:
        // Effects: constructs an empty future object that does not refer to
        //          an shared state.
        // Postcondition: valid() == false.
        unique_future() BOOST_NOEXCEPT
          : base_type()
        {}

        // Effects: move constructs a future object that refers to the shared
        //          state that was originally referred to by other (if any).
        // Postconditions:
        //   - valid() returns the same value as other.valid() prior to the
        //     constructor invocation.
        //   - other.valid() == false.
        unique_future(unique_future && other) BOOST_NOEXCEPT
          : base_type(std::move(other))
        {}

        // Effects: constructs a future object by moving the instance referred
        //          to by rhs and unwrapping the inner future (see unwrap()).
        // Postconditions:
        //   - valid() returns the same value as other.valid() prior to the
        //     constructor invocation.
        //   - other.valid() == false.
        unique_future(unique_future<unique_future> && other) BOOST_NOEXCEPT
          : base_type(std::move(other.unwrap()))
        {}

        // Effects:
        //   - releases any shared state (30.6.4);
        //   - destroys *this.
        ~unique_future()
        {}

        // [N3722, 4.1] asks for this...
        typedef lcos::local::promise<R> promise_type;
#ifdef BOOST_NO_CXX11_EXPLICIT_CONVERSION_OPERATORS
        // defined at promise.hpp
        explicit unique_future(promise_type& promise);
#endif

        // Effects:
        //   - releases any shared state (30.6.4).
        //   - move assigns the contents of other to *this.
        // Postconditions:
        //   - valid() returns the same value as other.valid() prior to the
        //     assignment.
        //   - other.valid() == false.
        unique_future& operator=(unique_future && other) BOOST_NOEXCEPT
        {
            base_type::operator=(std::move(other));
            return *this;
        }

        // Returns: shared_future<R>(std::move(*this)).
        // Postcondition: valid() == false.
        shared_future<R> share()
        {
            return shared_future<R>(std::move(*this));
        }

        // Effects: wait()s until the shared state is ready, then retrieves
        //          the value stored in the shared state.
        // Returns:
        //   - future::get() returns the value v stored in the object's
        //     shared state as std::move(v).
        //   - future<R&>::get() returns the reference stored as value in the
        //     object's shared state.
        //   - future<void>::get() returns nothing.
        // Throws: the stored exception, if an exception was stored in the
        //         shared state.
        // Postcondition: valid() == false.
        R get()
        {
            if (!this->shared_state_)
            {
                HPX_THROW_EXCEPTION(no_state,
                    "unique_future<R>::get",
                    "this future has no valid shared state");
            }

            invalidate on_exit(*this);

            typedef typename shared_state_type::data_type data_type;
            data_type& data = this->shared_state_->get_result();

            // no error has been reported, return the result
            return detail::future_value<R>::get(data.move_value());
        }
        R get(error_code& ec)
        {
            if (!this->shared_state_)
            {
                HPX_THROWS_IF(ec, no_state,
                    "unique_future<R>::get",
                    "this future has no valid shared state");
                return detail::future_value<R>::get_default();
            }

            invalidate on_exit(*this);

            typedef typename shared_state_type::data_type data_type;
            data_type& data = this->shared_state_->get_result(ec);
            if (ec) return detail::future_value<R>::get_default();

            // no error has been reported, return the result
            return detail::future_value<R>::get(data.move_value());
        }

        using base_type::valid;
        using base_type::is_ready;
        using base_type::has_value;
        using base_type::has_exception;
        using base_type::get_status;

        template <typename F>
        typename detail::future_then_result<unique_future, F>::type
        then(F && f)
        {
            invalidate on_exit(*this);
            return base_type::then(std::forward<F>(f));
        }

        template <typename F>
        typename detail::future_then_result<unique_future, F>::type
        then(BOOST_SCOPED_ENUM(launch) policy, F && f)
        {
            invalidate on_exit(*this);
            return base_type::then(policy, std::forward<F>(f));
        }

        template <typename F>
        typename detail::future_then_result<unique_future, F>::type
        then(threads::executor& sched, F && f)
        {
            invalidate on_exit(*this);
            return base_type::then(sched, std::forward<F>(f));
        }

        typename detail::future_unwrap_result<unique_future>::type
        unwrap(error_code& ec = throws)
        {
            invalidate on_exit(*this);
            return base_type::unwrap(ec);
        }

        using base_type::wait;
        using base_type::wait_for;
        using base_type::wait_until;
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename R>
    class shared_future : public detail::future_base<shared_future<R>, R>
    {
        typedef detail::future_base<shared_future<R>, R> base_type;

    public:
        typedef R result_type;
        typedef typename base_type::shared_state_type shared_state_type;

    private:
        friend struct detail::future_access;

        // Effects: constructs a future object from an shared state
        explicit shared_future(
            boost::intrusive_ptr<shared_state_type> const& state
        ) : base_type(state)
        {}

        explicit shared_future(
            boost::intrusive_ptr<shared_state_type> && state
        ) : base_type(std::move(state))
        {}

        template <typename SharedState>
        explicit shared_future(boost::intrusive_ptr<SharedState> const& state)
          : base_type(boost::static_pointer_cast<shared_state_type>(state))
        {}

    public:
        // Effects: constructs an empty future object that does not refer to
        //          an shared state.
        // Postcondition: valid() == false.
        shared_future() BOOST_NOEXCEPT
          : base_type()
        {}

        // Effects: constructs a shared_future object that refers to the same
        //          shared state as other (if any).
        // Postcondition: valid() returns the same value as other.valid().
        shared_future(shared_future const& other)
          : base_type(other)
        {}

        // Effects: move constructs a future object that refers to the shared
        //          state that was originally referred to by other (if any).
        // Postconditions:
        //   - valid() returns the same value as other.valid() prior to the
        //     constructor invocation.
        //   - other.valid() == false.
        shared_future(shared_future && other) BOOST_NOEXCEPT
          : base_type(std::move(other))
        {}

        shared_future(unique_future<R> && other) BOOST_NOEXCEPT
          : base_type(detail::future_access::get_shared_state(other))
        {
            other = unique_future<R>();
        }

        // Effects:
        //   - releases any shared state (30.6.4);
        //   - destroys *this.
        ~shared_future()
        {}

        // [N3722, 4.1] asks for this...
        typedef lcos::local::promise<R> promise_type;
#ifdef BOOST_NO_CXX11_EXPLICIT_CONVERSION_OPERATORS
        // defined at promise.hpp
        explicit shared_future(promise_type& promise);
#endif

        // Effects:
        //   - releases any shared state (30.6.4).
        //   - assigns the contents of other to *this. As a result, *this
        //     refers to the same shared state as other (if any).
        // Postconditions:
        //   - valid() == other.valid().
        shared_future& operator=(shared_future const & other)
        {
            base_type::operator=(other);
            return *this;
        }

        // Effects:
        //   - releases any shared state (30.6.4).
        //   - move assigns the contents of other to *this.
        // Postconditions:
        //   - valid() returns the same value as other.valid() prior to the
        //     assignment.
        //   - other.valid() == false.
        shared_future& operator=(shared_future && other) BOOST_NOEXCEPT
        {
            base_type::operator=(std::move(other));
            return *this;
        }

        // Effects: wait()s until the shared state is ready, then retrieves
        //          the value stored in the shared state.
        // Returns:
        //   - shared_future::get() returns a const reference to the value
        //     stored in the object's shared state.
        //   - shared_future<R&>::get() returns the reference stored as value
        //     in the object's shared state.
        //   - shared_future<void>::get() returns nothing.
        // Throws: the stored exception, if an exception was stored in the
        //         shared state.
        // Postcondition: valid() == false.
        typename detail::future_value<R>::const_lvref get() const
        {
            if (!this->shared_state_)
            {
                HPX_THROW_EXCEPTION(no_state,
                    "shared_future<R>::get",
                    "this future has no valid shared state");
            }

            typedef typename shared_state_type::data_type data_type;
            data_type& data = this->shared_state_->get_result();

            // no error has been reported, return the result
            return detail::future_value<R>::get(data.get_value());
        }
        typename detail::future_value<R>::const_lvref get(error_code& ec) const
        {
            if (!this->shared_state_)
            {
                HPX_THROWS_IF(ec, no_state,
                    "shared_future<R>::get",
                    "this future has no valid shared state");
                return detail::future_value<R>::get_default();
            }

            typedef typename shared_state_type::data_type data_type;
            data_type& data = this->shared_state_->get_result(ec);
            if (ec) return detail::future_value<R>::get_default();

            // no error has been reported, return the result
            return detail::future_value<R>::get(data.get_value());
        }

        using base_type::valid;
        using base_type::is_ready;
        using base_type::has_value;
        using base_type::has_exception;
        using base_type::get_status;

        using base_type::then;

        using base_type::unwrap;

        using base_type::wait;
        using base_type::wait_for;
        using base_type::wait_until;
    };

    ///////////////////////////////////////////////////////////////////////////
    namespace local { namespace detail
    {
        template <typename Policy>
        struct is_launch_policy
          : boost::mpl::or_<
                boost::is_same<BOOST_SCOPED_ENUM(launch), Policy>
              , boost::is_base_and_derived<threads::executor, Policy>
            >
        {};
    }}

#if defined(HPX_ENABLE_DEPRECATED_FUTURE)
    ///////////////////////////////////////////////////////////////////////////
    template <typename Result>
    class future
    {
    public:
        typedef lcos::detail::future_data<Result> future_data_type;

    private:
        friend struct detail::future_access;

        explicit future(
            boost::intrusive_ptr<future_data_type> const& future_data
        ) : future_data_(future_data)
        {}

        explicit future(
            boost::intrusive_ptr<future_data_type> && future_data
        ) : future_data_(std::move(future_data))
        {}

        template <typename U>
        explicit future(boost::intrusive_ptr<U> const& future_data)
          : future_data_(boost::static_pointer_cast<future_data_type>(future_data))
        {}

    public:
        typedef Result result_type;

        future()
        {}

        ~future()
        {}

        future(future const& other)
          : future_data_(other.future_data_)
        {
        }

        future(future && other)
          : future_data_(other.future_data_)
        {
            other.future_data_.reset();
        }

        // accept unique_future future
        future(unique_future<Result> && other)
          : future_data_(detail::future_access::get_shared_state(other))
        {
            other = unique_future<Result>();
        }

        // accept wrapped future
        future(future<future> && other)
        {
            future f = other.unwrap();
            future_data_ = std::move(f.future_data_);
        }

        future(unique_future<future> && other)
        {
            future f = other.unwrap();
            future_data_ = std::move(f.future_data_);
        }

        typedef lcos::local::promise<Result> promise_type;
#ifdef BOOST_NO_CXX11_EXPLICIT_CONVERSION_OPERATORS
        // [N3722, 4.1] asks for this... defined at promise.hpp
        explicit future(promise_type& promise);
#endif

        // assignment
        future& operator=(future const & other)
        {
            if (this != &other)
                future_data_ = other.future_data_;
            return *this;
        }

        future& operator=(future && other)
        {
            if (this != &other)
            {
                future_data_ = other.future_data_;
                other.future_data_.reset();
            }
            return *this;
        }

        void swap(future& other)
        {
            future_data_.swap(other.future_data_);
        }

        // retrieving the value
        Result const& get() const
        {
            if (!future_data_) {
                HPX_THROW_EXCEPTION(no_state,
                    "future<Result>::get",
                    "this future has no valid shared state");
            }

            typedef typename future_data_type::data_type data_type;
            data_type& data = future_data_->get_result();

            // no error has been reported, return the result
            return data.get_value();
        }

        Result const& get(error_code& ec) const
        {
            static result_type default_;

            if (!future_data_) {
                HPX_THROWS_IF(ec, no_state,
                    "future<Result>::get",
                    "this future has no valid shared state");
                return default_;
            }

            typedef typename future_data_type::data_type data_type;
            data_type& data = future_data_->get_result(ec);
            if (ec) return default_;

            // no error has been reported, return the result
            return data.get_value();
        }

    private:
        struct invalidate
        {
            invalidate(future& f)
              : f_(f)
            {}

            ~invalidate()
            {
                // This resets the intrusive pointer itself, not the future_data_.
                f_.future_data_.reset();
            }

            future& f_;
        };
        friend struct invalidate;

    public:
        Result move()
        {
            if (!future_data_) {
                HPX_THROW_EXCEPTION(no_state,
                    "future<Result>::move",
                    "this future has no valid shared state");
            }

            invalidate on_exit(*this);

            typedef typename future_data_type::data_type data_type;
            data_type& data = future_data_->get_result();

            // no error has been reported, return the result
            return data.move_value();
        }

        Result move(error_code& ec)
        {
            static result_type default_;

            if (!future_data_) {
                HPX_THROWS_IF(ec, no_state,
                    "future<Result>::move",
                    "this future has no valid shared state");
                return default_;
            }

            invalidate on_exit(*this);

            typedef typename future_data_type::data_type data_type;
            data_type& data = future_data_->get_result(ec);
            if (ec) return default_;

            // no error has been reported, return the result
            return data.move_value();
        }

        // state introspection
        bool is_ready() const
        {
            return future_data_ != 0 && future_data_->is_ready();
        }

        bool has_value() const
        {
            return future_data_ != 0 && future_data_->has_value();
        }

        bool has_exception() const
        {
            return future_data_ != 0 && future_data_->has_exception();
        }

        BOOST_SCOPED_ENUM(future_status) get_status() const
        {
            if (!future_data_)
                return future_status::uninitialized;

            return future_data_->get_status();
        }

        // cancellation support
        bool cancelable() const
        {
            return future_data_->cancelable();
        }

        void cancel()
        {
            future_data_->cancel();
        }

        bool valid() const BOOST_NOEXCEPT
        {
            // avoid warning about conversion to bool
            return future_data_.get() ? true : false;
        }

        // continuation support
        template <typename F>
        typename detail::future_then_result<future, F>::type
        then(F && f);

        template <typename F>
        typename detail::future_then_result<future, F>::type
        then(BOOST_SCOPED_ENUM(launch) policy, F && f);

        template <typename F>
        typename detail::future_then_result<future, F>::type
        then(threads::executor& sched, F && f);

        // wait support
        void wait() const
        {
            future_data_->wait();
        }

        BOOST_SCOPED_ENUM(future_status)
        wait_until(boost::posix_time::ptime const& at)
        {
            return future_data_->wait_until(at);
        }

        BOOST_SCOPED_ENUM(future_status)
        wait_for(boost::posix_time::time_duration const& p)
        {
            return future_data_->wait_for(p);
        }

        template <typename Clock, typename Duration>
        BOOST_SCOPED_ENUM(future_status)
        wait_until(boost::chrono::time_point<Clock, Duration> const& abs_time)
        {
            return wait_until(util::to_ptime(abs_time));
        }

        template <typename Rep, typename Period>
        BOOST_SCOPED_ENUM(future_status)
        wait_for(boost::chrono::duration<Rep, Period> const& rel_time)
        {
            return wait_for(util::to_time_duration(rel_time));
        }

        typename detail::future_unwrap_result<future>::type
        unwrap(error_code& ec = throws);

    protected:
        boost::intrusive_ptr<future_data_type> future_data_;
    };
#endif

    ///////////////////////////////////////////////////////////////////////////
    // extension: create a pre-initialized future object
    template <typename Result>
    unique_future<typename util::decay<Result>::type>
    make_ready_future(Result && init)
    {
        typedef typename util::decay<Result>::type result_type;
        typedef lcos::detail::future_data<result_type> shared_state;

        boost::intrusive_ptr<shared_state> p(new shared_state());
        p->set_data(std::forward<Result>(init));

        using lcos::detail::future_access;
        return future_access::create<unique_future<result_type> >(
            std::move(p));
    }

    // extension: create a pre-initialized future object which holds the
    // given error
    template <typename Result>
    unique_future<Result>
    make_error_future(boost::exception_ptr const& e)
    {
        typedef typename util::decay<Result>::type result_type;
        typedef lcos::detail::future_data<result_type> shared_state;

        boost::intrusive_ptr<shared_state> p(new shared_state());
        p->set_exception(e);

        using lcos::detail::future_access;
        return future_access::create<unique_future<result_type> >(
            std::move(p));
    }

    // extension: create a pre-initialized future object which gets ready at
    // a given point in time
    template <typename Result>
    unique_future<typename util::decay<Result>::type>
    make_ready_future_at(boost::posix_time::ptime const& at,
        Result && init)
    {
        typedef typename util::decay<Result>::type result_type;
        typedef lcos::detail::timed_future_data<result_type> shared_state;

        using lcos::detail::future_access;
        return future_access::create<unique_future<result_type> >(
            new shared_state(at, std::forward<Result>(init)));
    }

    template <typename Clock, typename Duration, typename Result>
    unique_future<typename util::decay<Result>::type>
    make_ready_future_at(boost::chrono::time_point<Clock, Duration> const& at,
        Result && init)
    {
        return make_ready_future_at(
            util::to_ptime(at), std::forward<Result>(init));
    }

    template <typename Result>
    unique_future<typename util::decay<Result>::type>
    make_ready_future_after(boost::posix_time::time_duration const& d,
        Result && init)
    {
        typedef typename util::decay<Result>::type result_type;
        typedef lcos::detail::timed_future_data<result_type> shared_state;

        using lcos::detail::future_access;
        return future_access::create<unique_future<result_type> >(
            new shared_state(d, std::forward<Result>(init)));
    }

    template <typename Rep, typename Period, typename Result>
    unique_future<typename util::decay<Result>::type>
    make_ready_future_after(boost::chrono::duration<Rep, Period> const& d,
        Result && init)
    {
        return make_ready_future_at(
            util::to_time_duration(d), std::forward<Result>(init));
    }

#if defined(HPX_ENABLE_DEPRECATED_FUTURE)
    ///////////////////////////////////////////////////////////////////////////
    template <>
    class future<void>
    {
    public:
        typedef lcos::detail::future_data<void> future_data_type;
        typedef future_data_type::data_type data_type;

    private:
        friend struct detail::future_access;

        template <typename U>
        explicit future(boost::intrusive_ptr<U> const& u)
          : future_data_(boost::static_pointer_cast<future_data_type>(u))
        {}

    public:
        typedef void result_type;

        future()
        {}

        ~future()
        {}

        future(future const& other)
          : future_data_(other.future_data_)
        {
        }

        future(future && other)
          : future_data_(other.future_data_)
        {
            other.future_data_.reset();
        }

        // accept unique_future future
        future(unique_future<void> && other)
          : future_data_(detail::future_access::get_shared_state(other))
        {
            other = unique_future<void>();
        }

        // extension: accept wrapped future
        future(future<future> && other)
        {
            future f = other.unwrap();
            future_data_ = std::move(f.future_data_);
        }

        future(unique_future<future> && other)
        {
            future f = other.unwrap();
            future_data_ = std::move(f.future_data_);
        }

        typedef lcos::local::promise<void> promise_type;
#ifdef BOOST_NO_CXX11_EXPLICIT_CONVERSION_OPERATORS
        // [N3722, 4.1] asks for this... defined at promise.hpp
        explicit future(promise_type& promise);
#endif

        // assignment
        future& operator=(future const & other)
        {
            if (this != &other)
                future_data_ = other.future_data_;
            return *this;
        }

        future& operator=(future && other)
        {
            if (this != &other)
            {
                future_data_ = other.future_data_;
                other.future_data_.reset();
            }
            return *this;
        }

        void swap(future& other)
        {
            future_data_.swap(other.future_data_);
        }

        // retrieving the value
        void get(error_code& ec = throws) const
        {
            if (!future_data_) {
                HPX_THROW_EXCEPTION(no_state,
                    "future<void>::get",
                    "this future has no valid shared state");
            }

            future_data_->get_result(ec);
        }

        void move(error_code& ec = throws)
        {
            if (!future_data_) {
                HPX_THROW_EXCEPTION(no_state,
                    "future<void>::get",
                    "this future has no valid shared state");
            }

            future_data_->get_result(ec);

            // This resets the intrusive pointer itself, not the future_data_
            future_data_.reset();
        }

        // state introspection
        bool is_ready() const
        {
            return future_data_ != 0 && future_data_->is_ready();
        }

        bool has_value() const
        {
            return future_data_ != 0 && future_data_->has_value();
        }

        bool has_exception() const
        {
            return future_data_ != 0 && future_data_->has_exception();
        }

        BOOST_SCOPED_ENUM(future_status) get_status() const
        {
            if (!future_data_)
                return future_status::uninitialized;

            return future_data_->get_status();
        }

        // cancellation support
        bool cancelable() const
        {
            return future_data_->cancelable();
        }

        void cancel()
        {
            future_data_->cancel();
        }

        bool valid() const BOOST_NOEXCEPT
        {
            // avoid warning about conversion to bool
            return future_data_.get() ? true : false;
        }

        // continuation support
        template <typename F>
        typename detail::future_then_result<future, F>::type
        then(F && f);

        template <typename F>
        typename detail::future_then_result<future, F>::type
        then(BOOST_SCOPED_ENUM(launch) policy, F && f);

        template <typename F>
        typename detail::future_then_result<future, F>::type
        then(threads::executor& sched, F && f);

        // wait support
        void wait() const
        {
            future_data_->wait();
        }

        BOOST_SCOPED_ENUM(future_status)
        wait_until(boost::posix_time::ptime const& at)
        {
            return future_data_->wait_until(at);
        }

        BOOST_SCOPED_ENUM(future_status)
        wait_for(boost::posix_time::time_duration const& p)
        {
            return future_data_->wait_for(p);
        }

        template <typename Clock, typename Duration>
        BOOST_SCOPED_ENUM(future_status)
        wait_until(boost::chrono::time_point<Clock, Duration> const& abs_time)
        {
            return wait_until(util::to_ptime(abs_time));
        }

        template <typename Rep, typename Period>
        BOOST_SCOPED_ENUM(future_status)
        wait_for(boost::chrono::duration<Rep, Period> const& rel_time)
        {
            return wait_for(util::to_time_duration(rel_time));
        }

    protected:
        boost::intrusive_ptr<future_data_type> future_data_;
    };
#endif

    // extension: create a pre-initialized future object
    inline unique_future<void> make_ready_future()
    {
        typedef lcos::detail::future_data<void> shared_state;

        boost::intrusive_ptr<shared_state> p(new shared_state());
        p->set_data(util::unused);

        using lcos::detail::future_access;
        return future_access::create<unique_future<void> >(std::move(p));
    }

    // extension: create a pre-initialized future object which gets ready at
    // a given point in time
    inline unique_future<void> make_ready_future_at(
        boost::posix_time::ptime const& at)
    {
        typedef lcos::detail::timed_future_data<void> shared_state;

        using lcos::detail::future_access;
        return future_access::create<unique_future<void> >(
            new shared_state(at, util::unused));
    }

    template <typename Clock, typename Duration>
    inline unique_future<void> make_ready_future_at(
        boost::chrono::time_point<Clock, Duration> const& at)
    {
        return make_ready_future_at(util::to_ptime(at));
    }

    inline unique_future<void> make_ready_future_after(
        boost::posix_time::time_duration const& d)
    {
        typedef lcos::detail::timed_future_data<void> shared_state;

        using lcos::detail::future_access;
        return future_access::create<unique_future<void> >(
            new shared_state(d, util::unused));
    }

    template <typename Rep, typename Period>
    inline unique_future<void> make_ready_future_at(
        boost::chrono::duration<Rep, Period> const& d)
    {
        return make_ready_future_after(util::to_time_duration(d));
    }
}}

#if defined(HPX_ENABLE_DEPRECATED_FUTURE)
HPX_REGISTER_TYPED_CONTINUATION_DECLARATION(
    hpx::lcos::future<void>,
    hpx_lcos_future_void_typed_continuation)
#endif

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace actions
{
    // special handling of actions returning a future
    template <typename R>
    struct typed_continuation<lcos::unique_future<R> > : continuation
    {
        typed_continuation()
        {}

        explicit typed_continuation(naming::id_type const& gid)
          : continuation(gid)
        {}
        explicit typed_continuation(naming::id_type && gid)
          : continuation(std::move(gid))
        {}

        template <typename F>
        explicit typed_continuation(naming::id_type const& gid,
                F && f)
          : continuation(gid), f_(std::forward<F>(f))
        {}
        template <typename F>
        explicit typed_continuation(naming::id_type && gid,
                F && f)
          : continuation(std::move(gid)), f_(std::forward<F>(f))
        {}

        template <typename F>
        explicit typed_continuation(F && f)
          : f_(std::forward<F>(f))
        {}

        ~typed_continuation()
        {
            init_registration<typed_continuation>::g.register_continuation();
        }

        void deferred_trigger(lcos::unique_future<R> result, boost::mpl::false_) const
        {
            if (f_.empty()) {
                if (!this->get_gid()) {
                    HPX_THROW_EXCEPTION(invalid_status,
                        "typed_continuation<lcos::unique_future<R> >::trigger_value",
                        "attempt to trigger invalid LCO (the id is invalid)");
                    return;
                }
                hpx::set_lco_value(this->get_gid(), result.get());
            }
            else {
                f_(this->get_gid(), result.get());
            }
        }

        void deferred_trigger(lcos::unique_future<R> result, boost::mpl::true_) const
        {
            if (f_.empty()) {
                if (!this->get_gid()) {
                    HPX_THROW_EXCEPTION(invalid_status,
                        "typed_continuation<lcos::unique_future<R> >::trigger_value",
                        "attempt to trigger invalid LCO (the id is invalid)");
                    return;
                }
                result.get();
                hpx::set_lco_value(this->get_gid());
            }
            else {
                result.get();
                f_(this->get_gid());
            }
        }

        void trigger_value(lcos::unique_future<R> && result) const
        {
            LLCO_(info)
                << "typed_continuation<lcos::unique_future<R> >::trigger("
                << this->get_gid() << ")";
            
            typedef boost::mpl::bool_<boost::is_void<R>::value> predicate;

            // if the future is ready, send the result back immediately
            if (result.is_ready()) {
                deferred_trigger(std::move(result), predicate());
                return;
            }

            // attach continuation to this future which will send the result back
            // once its ready
            void(typed_continuation::*f)(lcos::unique_future<R>, predicate) const
                = &typed_continuation::deferred_trigger;

            deferred_result_ = result.then(
                util::bind(f,
                    boost::static_pointer_cast<typed_continuation const>(
                        shared_from_this()),
                    util::placeholders::_1, predicate()));
        }

    private:
        char const* get_continuation_name() const
        {
            return detail::get_continuation_name<typed_continuation>();
        }

        /// serialization support
        void load(hpx::util::portable_binary_iarchive& ar)
        {
            // serialize base class
            typedef continuation base_type;
            this->base_type::load(ar);

            // serialize function
            bool have_function = false;
            ar.load(have_function);
            if (have_function)
                ar >> f_;
        }
        void save(hpx::util::portable_binary_oarchive& ar) const
        {
            // serialize base class
            typedef continuation base_type;
            this->base_type::save(ar);

            // serialize function
            bool have_function = !f_.empty();
            ar.save(have_function);
            if (have_function)
                ar << f_;
        }

        util::function<void(naming::id_type, R)> f_;
        mutable lcos::unique_future<void> deferred_result_;
    };

    template <typename R>
    struct typed_continuation<lcos::shared_future<R> > : continuation
    {
        typed_continuation()
        {}

        explicit typed_continuation(naming::id_type const& gid)
          : continuation(gid)
        {}
        explicit typed_continuation(naming::id_type && gid)
          : continuation(std::move(gid))
        {}

        template <typename F>
        explicit typed_continuation(naming::id_type const& gid,
                F && f)
          : continuation(gid), f_(std::forward<F>(f))
        {}
        template <typename F>
        explicit typed_continuation(naming::id_type && gid,
                F && f)
          : continuation(std::move(gid)), f_(std::forward<F>(f))
        {}

        template <typename F>
        explicit typed_continuation(F && f)
          : f_(std::forward<F>(f))
        {}

        ~typed_continuation()
        {
            init_registration<typed_continuation>::g.register_continuation();
        }

        void deferred_trigger(lcos::shared_future<R> result, boost::mpl::false_) const
        {
            if (f_.empty()) {
                if (!this->get_gid()) {
                    HPX_THROW_EXCEPTION(invalid_status,
                        "typed_continuation<lcos::shared_future<R> >::trigger_value",
                        "attempt to trigger invalid LCO (the id is invalid)");
                    return;
                }
                hpx::set_lco_value(this->get_gid(), result.get());
            }
            else {
                f_(this->get_gid(), result.get());
            }
        }

        void deferred_trigger(lcos::shared_future<R> result, boost::mpl::true_) const
        {
            if (f_.empty()) {
                if (!this->get_gid()) {
                    HPX_THROW_EXCEPTION(invalid_status,
                        "typed_continuation<lcos::shared_future<R> >::trigger_value",
                        "attempt to trigger invalid LCO (the id is invalid)");
                    return;
                }
                result.get();
                hpx::set_lco_value(this->get_gid());
            }
            else {
                result.get();
                f_(this->get_gid());
            }
        }

        void trigger_value(lcos::shared_future<R> && result) const
        {
            LLCO_(info)
                << "typed_continuation<lcos::shared_future<R> >::trigger("
                << this->get_gid() << ")";
            
            typedef boost::mpl::bool_<boost::is_void<R>::value> predicate;

            // if the future is ready, send the result back immediately
            if (result.is_ready()) {
                deferred_trigger(std::move(result), predicate());
                return;
            }

            // attach continuation to this future which will send the result back
            // once its ready
            void(typed_continuation::*f)(lcos::shared_future<R>, predicate) const
                = &typed_continuation::deferred_trigger;

            deferred_result_ = result.then(
                util::bind(f,
                    boost::static_pointer_cast<typed_continuation const>(
                        shared_from_this()),
                    util::placeholders::_1, predicate()));
        }

    private:
        char const* get_continuation_name() const
        {
            return detail::get_continuation_name<typed_continuation>();
        }

        /// serialization support
        void load(hpx::util::portable_binary_iarchive& ar)
        {
            // serialize base class
            typedef continuation base_type;
            this->base_type::load(ar);

            // serialize function
            bool have_function = false;
            ar.load(have_function);
            if (have_function)
                ar >> f_;
        }
        void save(hpx::util::portable_binary_oarchive& ar) const
        {
            // serialize base class
            typedef continuation base_type;
            this->base_type::save(ar);

            // serialize function
            bool have_function = !f_.empty();
            ar.save(have_function);
            if (have_function)
                ar << f_;
        }

        util::function<void(naming::id_type, R)> f_;
        mutable lcos::shared_future<void> deferred_result_;
    };

#if defined(HPX_ENABLE_DEPRECATED_FUTURE)
    template <typename Result>
    struct typed_continuation<lcos::future<Result> > : continuation
    {
        typed_continuation()
        {}

        explicit typed_continuation(naming::id_type const& gid)
          : continuation(gid)
        {}

        explicit typed_continuation(naming::id_type && gid)
          : continuation(std::move(gid))
        {}

        template <typename F>
        explicit typed_continuation(naming::id_type const& gid,
                F && f)
          : continuation(gid), f_(std::forward<F>(f))
        {}

        template <typename F>
        explicit typed_continuation(naming::id_type && gid,
                F && f)
          : continuation(std::move(gid)), f_(std::forward<F>(f))
        {}

        template <typename F>
        explicit typed_continuation(F && f)
          : f_(std::forward<F>(f))
        {}

        ~typed_continuation()
        {
            init_registration<typed_continuation>::g.register_continuation();
        }

        void deferred_trigger(lcos::future<Result> result) const
        {
            if (f_.empty()) {
                if (!this->get_gid()) {
                    HPX_THROW_EXCEPTION(invalid_status,
                        "typed_continuation<lcos::future<Result> >::trigger_value",
                        "attempt to trigger invalid LCO (the id is invalid)");
                    return;
                }
                hpx::set_lco_value(this->get_gid(), result.get());
            }
            else {
                f_(this->get_gid(), result.get());
            }
        }

        void trigger_value(lcos::future<Result> && result) const
        {
            LLCO_(info)
                << "typed_continuation<lcos::future<hpx::lcos::future<Result> > >::trigger("
                << this->get_gid() << ")";

            // if the future is ready, send the result back immediately
            if (result.is_ready()) {
                deferred_trigger(result);
                return;
            }

            // attach continuation to this future which will send the result back
            // once its ready
            deferred_result_ = result.then(
                util::bind(&typed_continuation::deferred_trigger,
                    boost::static_pointer_cast<typed_continuation const>(
                        shared_from_this()),
                    util::placeholders::_1));
        }

    private:
        char const* get_continuation_name() const
        {
            return detail::get_continuation_name<typed_continuation>();
        }

        /// serialization support
        void load(hpx::util::portable_binary_iarchive& ar)
        {
            // serialize base class
            typedef continuation base_type;
            this->base_type::load(ar);

            // serialize function
            bool have_function = false;
            ar.load(have_function);
            if (have_function)
                ar >> f_;
        }
        void save(hpx::util::portable_binary_oarchive& ar) const
        {
            // serialize base class
            typedef continuation base_type;
            this->base_type::save(ar);

            // serialize function
            bool have_function = !f_.empty();
            ar.save(have_function);
            if (have_function)
                ar << f_;
        }

        util::function<void(naming::id_type, Result)> f_;
        mutable lcos::future<void> deferred_result_;
    };

    // special handling of actions returning a future
    template <>
    struct typed_continuation<lcos::future<void> > : continuation
    {
        typed_continuation()
        {}

        explicit typed_continuation(naming::id_type const& gid)
          : continuation(gid)
        {}

        explicit typed_continuation(naming::id_type && gid)
          : continuation(std::move(gid))
        {}

        template <typename F>
        explicit typed_continuation(naming::id_type const& gid,
                F && f)
          : continuation(gid), f_(std::forward<F>(f))
        {}

        template <typename F>
        explicit typed_continuation(naming::id_type && gid,
                F && f)
          : continuation(std::move(gid)), f_(std::forward<F>(f))
        {}

        template <typename F>
        explicit typed_continuation(F && f)
          : f_(std::forward<F>(f))
        {}

        ~typed_continuation()
        {
            init_registration<typed_continuation>::g.register_continuation();
        }

        void deferred_trigger(lcos::future<void> result) const
        {
            if (f_.empty()) {
                if (!this->get_gid()) {
                    HPX_THROW_EXCEPTION(invalid_status,
                        "typed_continuation<lcos::future<void> >::trigger_value",
                        "attempt to trigger invalid LCO (the id is invalid)");
                    return;
                }
                result.get();
                hpx::trigger_lco_event(this->get_gid());
            }
            else {
                result.get();
                f_(this->get_gid());
            }
        }

        void trigger_value(lcos::future<void> && result) const
        {
            LLCO_(info)
                << "typed_continuation<lcos::future<hpx::lcos::future<void> > >::trigger("
                << this->get_gid() << ")";

            // if the future is ready, send the result back immediately
            if (result.is_ready()) {
                deferred_trigger(result);
                return;
            }

            // attach continuation to this future which will send the result back
            // once its ready
            deferred_result_ = result.then(
                util::bind(&typed_continuation::deferred_trigger,
                    boost::static_pointer_cast<typed_continuation const>(
                        shared_from_this()),
                    util::placeholders::_1));
        }

    private:
        char const* get_continuation_name() const
        {
            return detail::get_continuation_name<typed_continuation>();
        }

        /// serialization support
        void load(hpx::util::portable_binary_iarchive& ar)
        {
            // serialize base class
            typedef continuation base_type;
            this->base_type::load(ar);

            // serialize function
            bool have_function = false;
            ar.load(have_function);
            if (have_function)
                ar >> f_;
        }
        void save(hpx::util::portable_binary_oarchive& ar) const
        {
            // serialize base class
            typedef continuation base_type;
            this->base_type::save(ar);

            // serialize function
            bool have_function = !f_.empty();
            ar.save(have_function);
            if (have_function)
                ar << f_;
        }

        util::function<void(naming::id_type)> f_;
        mutable lcos::future<void> deferred_result_;
    };
#endif
}}

///////////////////////////////////////////////////////////////////////////////
// hoist names into main namespace
namespace hpx
{
    using lcos::make_ready_future;
    using lcos::make_error_future;
    using lcos::make_ready_future_at;
    using lcos::make_ready_future_after;
}

#define HPX_MAKE_ERROR_FUTURE(T, errorcode, f, msg)                    \
    lcos::make_error_future<T>(HPX_GET_EXCEPTION(errorcode, f, msg))   \
    /**/

#endif
