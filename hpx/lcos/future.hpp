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
#include <hpx/traits/future_traits.hpp>
#include <hpx/traits/future_unwrap_getter.hpp>
#include <hpx/traits/is_launch_policy.hpp>
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
#include <boost/utility/enable_if.hpp>

namespace hpx { namespace lcos { namespace detail
{
    ///////////////////////////////////////////////////////////////////////////
    template <typename R>
    struct shared_state_ptr
    {
        typedef boost::intrusive_ptr<future_data<R> > type;
    };

    template <typename Future>
    struct shared_state_ptr_for
      : shared_state_ptr<typename traits::future_traits<Future>::type>
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

    ///////////////////////////////////////////////////////////////////////////
    template <typename Archive, typename Future>
    void serialize_future_load(Archive& ar, Future& f);

    template <typename Archive, typename Future>
    void serialize_future_save(Archive& ar, Future& f);
}}}

namespace hpx { namespace traits
{
    ///////////////////////////////////////////////////////////////////////////
    template <typename Future, typename Enable = void>
    struct future_access;

    template <typename R>
    struct future_access<future<R> >
    {
        template <typename SharedState>
        static future<R>
        create(boost::intrusive_ptr<SharedState> const& shared_state)
        {
            return future<R>(shared_state);
        }

        template <typename SharedState>
        static future<R>
        create(boost::intrusive_ptr<SharedState> && shared_state)
        {
            return future<R>(std::move(shared_state));
        }

        template <typename SharedState>
        static future<R>
        create(SharedState* shared_state)
        {
            return future<R>(boost::intrusive_ptr<SharedState>(shared_state));
        }

        BOOST_FORCEINLINE static
        typename lcos::detail::shared_state_ptr<R>::type const&
        get_shared_state(future<R> const& f)
        {
            return f.shared_state_;
        }

        template <typename Archive>
        static void load(Archive& ar, future<R>& f)
        {
            lcos::detail::serialize_future_load(ar, f);
        }

        template <typename Archive>
        static void save(Archive& ar, future<R>& f)
        {
            lcos::detail::serialize_future_save(ar, f);
        }
    };

    template <typename R>
    struct future_access<shared_future<R> >
    {
        template <typename SharedState>
        static shared_future<R>
        create(boost::intrusive_ptr<SharedState> const& shared_state)
        {
            return shared_future<R>(shared_state);
        }

        template <typename SharedState>
        static shared_future<R>
        create(boost::intrusive_ptr<SharedState> && shared_state)
        {
            return shared_future<R>(std::move(shared_state));
        }

        template <typename SharedState>
        static shared_future<R>
        create(SharedState* shared_state)
        {
            return shared_future<R>(boost::intrusive_ptr<SharedState>(shared_state));
        }

        BOOST_FORCEINLINE static
        typename lcos::detail::shared_state_ptr<R>::type const&
        get_shared_state(shared_future<R> const& f)
        {
            return f.shared_state_;
        }

        template <typename Archive>
        static void load(Archive& ar, shared_future<R>& f)
        {
            lcos::detail::serialize_future_load(ar, f);
        }

        template <typename Archive>
        static void save(Archive& ar, shared_future<R> const& f)
        {
            lcos::detail::serialize_future_save(ar, f);
        }
    };
}}

namespace hpx { namespace lcos { namespace detail
{
    template <typename Future>
    BOOST_FORCEINLINE
    typename shared_state_ptr<
        typename traits::future_traits<Future>::type
    >::type const&
    get_shared_state(Future const& f)
    {
        return traits::future_access<Future>::get_shared_state(f);
    }

    ///////////////////////////////////////////////////////////////////////////
    enum future_state
    {
        invalid = 0,
        has_value = 1,
        has_exception = 2
    };

    template <typename Archive, typename Future>
    void serialize_future_load(Archive& ar, Future& f)
    {
        typedef typename traits::future_traits<Future>::type value_type;
        typedef lcos::detail::future_data<value_type> shared_state;

        int state = future_state::invalid;
        ar >> state;
        if (state == future_state::has_value)
        {
            value_type value;
            ar >> value;

            boost::intrusive_ptr<shared_state> p(new shared_state());
            p->set_result(std::move(value));

            f = traits::future_access<Future>::create(std::move(p));
        } else if (state == future_state::has_exception) {
            boost::exception_ptr exception;
            ar >> exception;

            boost::intrusive_ptr<shared_state> p(new shared_state());
            p->set_exception(exception);

            f = traits::future_access<Future>::create(std::move(p));
        } else if (state == future_state::invalid) {
            f = Future();
        } else {
            HPX_ASSERT(false);
        }
    }

    template <typename Archive, typename Future>
    void serialize_future_save(Archive& ar, Future& f)
    {
        HPX_ASSERT(!f.valid() || f.is_ready());
        
        int state = future_state::invalid;
        if (f.has_value())
        {
            state = future_state::has_value;
            ar << state << f.get();
        } else if (f.has_exception()) {
            state = future_state::has_exception;
            ar << state; //! fix me: use get_exception
            try
            {
                f.get();
            } catch (...) {
                boost::exception_ptr exception = boost::current_exception();
                ar << exception;
            }
        } else {
            state = future_state::invalid;
            ar << state;
        }
    }

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
        typedef typename util::result_of<F(Future)>::type cont_result;

        typedef typename boost::mpl::eval_if<
            traits::is_future<cont_result>
          , traits::future_traits<cont_result>
          , boost::mpl::identity<cont_result>
        >::type result_type;

        typedef lcos::future<result_type> type;
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename Future, typename Enable = void>
    struct future_unwrap_result
    {
        typedef typename traits::future_traits<Future>::type outer_result;

        typedef lcos::future<
            typename boost::mpl::eval_if<
                traits::is_future<outer_result>
              , traits::future_traits<outer_result>
              , boost::mpl::identity<void>
            >::type> type;
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename Iter>
    struct future_iterator_traits
    {
        typedef
            typename boost::detail::iterator_traits<Iter>::value_type
            type;

        typedef traits::future_traits<type> traits_type;
    };

    template <typename T>
    struct future_iterator_traits<future<T> >
    {};

    template <typename T>
    struct future_iterator_traits<shared_future<T> >
    {};

    ///////////////////////////////////////////////////////////////////////////
    template <typename T>
    struct future_value
      : future_data_result<T>
    {
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
      : future_data_result<T&>
    {
        typedef T& const_lvref;

        BOOST_FORCEINLINE static
        T& get(T& u)
        {
            return u;
        }

        static T& get_default()
        {
            static T default_;
            return default_;
        }
    };

    template <>
    struct future_value<void>
      : future_data_result<void>
    {
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

    template <typename ContResult>
    struct continuation_result;

    template <typename ContResult, typename Future, typename F>
    inline typename shared_state_ptr<
        typename continuation_result<ContResult>::type
    >::type
    make_continuation(Future& future, BOOST_SCOPED_ENUM(launch) policy,
        F && f);

    template <typename ContResult, typename Future, typename F>
    inline typename shared_state_ptr<
        typename continuation_result<ContResult>::type
    >::type
    make_continuation(Future& future, threads::executor& sched,
        F && f);

    ///////////////////////////////////////////////////////////////////////////
    class void_continuation;

    template <typename Future>
    inline typename shared_state_ptr<void>::type
    make_void_continuation(Future& future);

    ///////////////////////////////////////////////////////////////////////////
    template <typename Future>
    struct unwrap_result
    {
        typedef typename traits::future_traits<Future>::type outer_result;

        typedef
            typename boost::mpl::eval_if<
                traits::is_future<outer_result>
              , traits::future_traits<outer_result>
              , boost::mpl::identity<void>
            >::type
            type;
    };

    template <typename Future>
    typename shared_state_ptr<
        typename unwrap_result<Future>::type>::type
    unwrap(Future& future, error_code& ec = throws);

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
        typename boost::lazy_disable_if<
            traits::is_launch_policy<
                typename util::decay<F>::type
            >
          , future_then_result<Derived, F>
        >::type
        then(F && f, error_code& ec = throws)
        {
            return then(launch::all, std::forward<F>(f), ec);
        }

        template <typename F>
        typename future_then_result<Derived, F>::type
        then(BOOST_SCOPED_ENUM(launch) policy, F && f, error_code& ec = throws)
        {
            typedef
                typename future_then_result<Derived, F>::result_type
                result_type;

            if (!shared_state_)
            {
                HPX_THROWS_IF(ec, no_state,
                    "future_base<R>::then",
                    "this future has no valid shared state");
                return future<result_type>();
            }

            typedef
                typename util::result_of<F(Derived)>::type
                continuation_result_type;
            typedef
                typename shared_state_ptr<result_type>::type
                shared_state_ptr;

            shared_state_ptr p =
                detail::make_continuation<continuation_result_type>(
                    *static_cast<Derived*>(this), policy, std::forward<F>(f));
            return traits::future_access<future<result_type> >::create(std::move(p));
        }

        template <typename F>
        typename future_then_result<Derived, F>::type
        then(threads::executor& sched, F && f, error_code& ec = throws)
        {
            typedef
                typename future_then_result<Derived, F>::result_type
                result_type;

            if (!shared_state_)
            {
                HPX_THROWS_IF(ec, no_state,
                    "future_base<R>::then",
                    "this future has no valid shared state");
                return future<result_type>();
            }

            typedef
                typename util::result_of<F(Derived)>::type
                continuation_result_type;
            typedef
                typename shared_state_ptr<result_type>::type
                shared_state_ptr;

            shared_state_ptr p =
                detail::make_continuation<continuation_result_type>(
                    *static_cast<Derived*>(this), sched, std::forward<F>(f));
            return traits::future_access<future<result_type> >::create(std::move(p));
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
        template <typename Getter>
        typename future_unwrap_result<Derived>::type
        unwrap(Getter getter, error_code& ec = throws)
        {
            return then(getter, ec);
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
    class future : public detail::future_base<future<R>, R>
    {
        HPX_MOVABLE_BUT_NOT_COPYABLE(future);

        typedef detail::future_base<future<R>, R> base_type;

    public:
        typedef R result_type;
        typedef typename base_type::shared_state_type shared_state_type;

    private:
        struct invalidate
        {
            explicit invalidate(future& f, bool reset_data = false)
              : f_(f), reset_data_(reset_data)
            {}

            ~invalidate()
            {
                if (reset_data_)
                    f_.shared_state_->reset();
                f_.shared_state_ = 0;
            }

            future& f_;
            bool reset_data_;
        };

    private:
        template <typename Future, typename Enable>
        friend struct traits::future_access;

        // Effects: constructs a future object from an shared state
        explicit future(
            boost::intrusive_ptr<shared_state_type> const& state
        ) : base_type(state)
        {}

        explicit future(
            boost::intrusive_ptr<shared_state_type> && state
        ) : base_type(std::move(state))
        {}

        template <typename SharedState>
        explicit future(boost::intrusive_ptr<SharedState> const& state)
          : base_type(boost::static_pointer_cast<shared_state_type>(state))
        {}

    public:
        // Effects: constructs an empty future object that does not refer to
        //          an shared state.
        // Postcondition: valid() == false.
        future() BOOST_NOEXCEPT
          : base_type()
        {}

        // Effects: move constructs a future object that refers to the shared
        //          state that was originally referred to by other (if any).
        // Postconditions:
        //   - valid() returns the same value as other.valid() prior to the
        //     constructor invocation.
        //   - other.valid() == false.
        future(future && other) BOOST_NOEXCEPT
          : base_type(std::move(other))
        {}

        // Effects: constructs a future object by moving the instance referred
        //          to by rhs and unwrapping the inner future (see unwrap()).
        // Postconditions:
        //   - valid() returns the same value as other.valid() prior to the
        //     constructor invocation.
        //   - other.valid() == false.
        future(future<future> && other) BOOST_NOEXCEPT
          : base_type(other.valid() ? other.unwrap() : base_type())
        {}

        // Effects: constructs a future<void> object that will be ready when
        //          the given future is ready
        // Postconditions:
        //   - valid() returns the same value as other.valid() prior to the
        //     constructor invocation.
        //   - other.valid() == false.
        template <typename T>
        future(future<T>&& other,
            typename boost::enable_if<boost::is_void<R>, T>::type* = 0
        ) : base_type(other.valid() ? detail::make_void_continuation(other) : 0)
        {
            other = future<T>();
        }

        // Effects:
        //   - releases any shared state (30.6.4);
        //   - destroys *this.
        ~future()
        {}

        // [N3722, 4.1] asks for this...
        typedef lcos::local::promise<R> promise_type;
#ifdef BOOST_NO_CXX11_EXPLICIT_CONVERSION_OPERATORS
        // defined at promise.hpp
        explicit future(promise_type& promise);
#endif

        // Effects:
        //   - releases any shared state (30.6.4).
        //   - move assigns the contents of other to *this.
        // Postconditions:
        //   - valid() returns the same value as other.valid() prior to the
        //     assignment.
        //   - other.valid() == false.
        future& operator=(future && other) BOOST_NOEXCEPT
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
                    "future<R>::get",
                    "this future has no valid shared state");
            }

            invalidate on_exit(*this, true);

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
                    "future<R>::get",
                    "this future has no valid shared state");
                return detail::future_value<R>::get_default();
            }

            invalidate on_exit(*this, true);

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
        typename boost::lazy_disable_if<
            traits::is_launch_policy<
                typename util::decay<F>::type
            >
          , detail::future_then_result<future, F>
        >::type
        then(F && f, error_code& ec = throws)
        {
            invalidate on_exit(*this);
            return base_type::then(std::forward<F>(f), ec);
        }

        template <typename F>
        typename detail::future_then_result<future, F>::type
        then(BOOST_SCOPED_ENUM(launch) policy, F && f, error_code& ec = throws)
        {
            invalidate on_exit(*this);
            return base_type::then(policy, std::forward<F>(f), ec);
        }

        template <typename F>
        typename detail::future_then_result<future, F>::type
        then(threads::executor& sched, F && f, error_code& ec = throws)
        {
            invalidate on_exit(*this);
            return base_type::then(sched, std::forward<F>(f), ec);
        }

        typename detail::future_unwrap_result<future>::type
        unwrap(error_code& ec = throws)
        {
            invalidate on_exit(*this);
            return base_type::unwrap(traits::future_unwrap_getter<future>(), ec);
        }

        using base_type::wait;
        using base_type::wait_for;
        using base_type::wait_until;
    };

    ///////////////////////////////////////////////////////////////////////////
    // allow to convert any future into a future<void>
    template <typename R>
    future<void> make_future_void(future<R>&& f)
    {
        return std::move(f);
    }
}}

HPX_REGISTER_TYPED_CONTINUATION_DECLARATION(
    hpx::lcos::future<void>,
    hpx_lcos_future_void_typed_continuation)

namespace hpx { namespace lcos
{
    ///////////////////////////////////////////////////////////////////////////
    template <typename R>
    class shared_future : public detail::future_base<shared_future<R>, R>
    {
        typedef detail::future_base<shared_future<R>, R> base_type;

    public:
        typedef R result_type;
        typedef typename base_type::shared_state_type shared_state_type;

    private:
        template <typename Future, typename Enable>
        friend struct traits::future_access;

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

        shared_future(future<R> && other) BOOST_NOEXCEPT
          : base_type(lcos::detail::get_shared_state(other))
        {
            other = future<R>();
        }

        // Effects: constructs a future<void> object that will be ready when
        //          the given future is ready
        // Postconditions:
        //   - valid() returns the same value as other.valid() prior to the
        //     constructor invocation.
        template <typename T>
        shared_future(shared_future<T> const& other,
            typename boost::enable_if<boost::is_void<R>, T>::type* = 0
        ) : base_type(other.valid() ? detail::make_void_continuation(other) : 0)
        {}

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

        typename detail::future_unwrap_result<shared_future>::type
        unwrap(error_code& ec = throws)
        {
            return base_type::unwrap(traits::future_unwrap_getter<shared_future>(), ec);
        }

        using base_type::wait;
        using base_type::wait_for;
        using base_type::wait_until;
    };

    ///////////////////////////////////////////////////////////////////////////
    // allow to convert any future into a future<void>
    template <typename R>
    shared_future<void> make_future_void(shared_future<R> const& f)
    {
        return f;
    }
}}

HPX_REGISTER_TYPED_CONTINUATION_DECLARATION(
    hpx::lcos::shared_future<void>,
    hpx_lcos_shared_future_void_typed_continuation)

namespace hpx { namespace lcos
{
    ///////////////////////////////////////////////////////////////////////////
    // extension: create a pre-initialized future object
    template <typename Result>
    future<typename util::detail::decay_unwrap<Result>::type>
    make_ready_future(Result && init)
    {
        typedef typename util::detail::decay_unwrap<Result>::type result_type;
        typedef lcos::detail::future_data<result_type> shared_state;

        boost::intrusive_ptr<shared_state> p(new shared_state());
        p->set_result(std::forward<Result>(init));

        using traits::future_access;
        return future_access<future<result_type> >::create(std::move(p));
    }

    // extension: create a pre-initialized future object which holds the
    // given error
    template <typename Result>
    future<typename util::detail::decay_unwrap<Result>::type>
    make_error_future(boost::exception_ptr const& e)
    {
        typedef typename util::detail::decay_unwrap<Result>::type result_type;
        typedef lcos::detail::future_data<result_type> shared_state;

        boost::intrusive_ptr<shared_state> p(new shared_state());
        p->set_exception(e);

        using traits::future_access;
        return future_access<future<result_type> >::create(std::move(p));
    }

    // extension: create a pre-initialized future object which gets ready at
    // a given point in time
    template <typename Result>
    future<typename util::detail::decay_unwrap<Result>::type>
    make_ready_future_at(boost::posix_time::ptime const& at,
        Result && init)
    {
        typedef typename util::detail::decay_unwrap<Result>::type result_type;
        typedef lcos::detail::timed_future_data<result_type> shared_state;

        using traits::future_access;
        return future_access<future<result_type> >::create(
            new shared_state(at, std::forward<Result>(init)));
    }

    template <typename Clock, typename Duration, typename Result>
    future<typename util::detail::decay_unwrap<Result>::type>
    make_ready_future_at(boost::chrono::time_point<Clock, Duration> const& at,
        Result && init)
    {
        return make_ready_future_at(
            util::to_ptime(at), std::forward<Result>(init));
    }

    template <typename Result>
    future<typename util::detail::decay_unwrap<Result>::type>
    make_ready_future_after(boost::posix_time::time_duration const& d,
        Result && init)
    {
        typedef typename util::detail::decay_unwrap<Result>::type result_type;
        typedef lcos::detail::timed_future_data<result_type> shared_state;

        using traits::future_access;
        return future_access<future<result_type> >::create(
            new shared_state(d, std::forward<Result>(init)));
    }

    template <typename Rep, typename Period, typename Result>
    future<typename util::detail::decay_unwrap<Result>::type>
    make_ready_future_after(boost::chrono::duration<Rep, Period> const& d,
        Result && init)
    {
        return make_ready_future_at(
            util::to_time_duration(d), std::forward<Result>(init));
    }

    // extension: create a pre-initialized future object
    inline future<void> make_ready_future()
    {
        typedef lcos::detail::future_data<void> shared_state;

        boost::intrusive_ptr<shared_state> p(new shared_state());
        p->set_result(util::unused);

        using traits::future_access;
        return future_access<future<void> >::create(std::move(p));
    }

    // extension: create a pre-initialized future object which gets ready at
    // a given point in time
    inline future<void> make_ready_future_at(
        boost::posix_time::ptime const& at)
    {
        typedef lcos::detail::timed_future_data<void> shared_state;

        using traits::future_access;
        return future_access<future<void> >::create(
            new shared_state(at, util::unused));
    }

    template <typename Clock, typename Duration>
    inline future<void> make_ready_future_at(
        boost::chrono::time_point<Clock, Duration> const& at)
    {
        return make_ready_future_at(util::to_ptime(at));
    }

    inline future<void> make_ready_future_after(
        boost::posix_time::time_duration const& d)
    {
        typedef lcos::detail::timed_future_data<void> shared_state;

        using traits::future_access;
        return future_access<future<void> >::create(
            new shared_state(d, util::unused));
    }

    template <typename Rep, typename Period>
    inline future<void> make_ready_future_at(
        boost::chrono::duration<Rep, Period> const& d)
    {
        return make_ready_future_after(util::to_time_duration(d));
    }
}}

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace actions
{
    // special handling of actions returning a future
    template <typename R>
    struct typed_continuation<lcos::future<R> > : continuation
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

        void deferred_trigger(lcos::future<R> result) const
        {
            if (f_.empty()) {
                if (!this->get_gid()) {
                    HPX_THROW_EXCEPTION(invalid_status,
                        "typed_continuation<lcos::future<R> >::trigger_value",
                        "attempt to trigger invalid LCO (the id is invalid)");
                    return;
                }
                hpx::set_lco_value(this->get_gid(), result.get());
            }
            else {
                f_(this->get_gid(), result.get());
            }
        }

        virtual void trigger_value(lcos::future<R> && result) const
        {
            LLCO_(info)
                << "typed_continuation<lcos::future<R> >::trigger("
                << this->get_gid() << ")";

            // if the future is ready, send the result back immediately
            if (result.is_ready()) {
                deferred_trigger(std::move(result));
                return;
            }

            // attach continuation to this future which will send the result back
            // once its ready
            result.then(
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

        util::function<void(naming::id_type, R)> f_;
    };

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

        virtual void trigger_value(lcos::future<void> && result) const
        {
            LLCO_(info)
                << "typed_continuation<lcos::future<void> >::trigger("
                << this->get_gid() << ")";

            // if the future is ready, send the result back immediately
            if (result.is_ready()) {
                deferred_trigger(std::move(result));
                return;
            }

            // attach continuation to this future which will send the result back
            // once its ready
            result.then(
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

        void deferred_trigger(lcos::shared_future<R> result) const
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

        void trigger_value(lcos::shared_future<R> && result) const
        {
            LLCO_(info)
                << "typed_continuation<lcos::shared_future<R> >::trigger("
                << this->get_gid() << ")";

            // if the future is ready, send the result back immediately
            if (result.is_ready()) {
                deferred_trigger(std::move(result));
                return;
            }

            // attach continuation to this future which will send the result back
            // once its ready
            result.then(
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

        util::function<void(naming::id_type, R)> f_;
    };

    template <>
    struct typed_continuation<lcos::shared_future<void> > : continuation
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

        void deferred_trigger(lcos::shared_future<void> result) const
        {
            if (f_.empty()) {
                if (!this->get_gid()) {
                    HPX_THROW_EXCEPTION(invalid_status,
                        "typed_continuation<lcos::shared_future<void> >::trigger_value",
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

        void trigger_value(lcos::shared_future<void> && result) const
        {
            LLCO_(info)
                << "typed_continuation<lcos::shared_future<R> >::trigger("
                << this->get_gid() << ")";

            // if the future is ready, send the result back immediately
            if (result.is_ready()) {
                deferred_trigger(std::move(result));
                return;
            }

            // attach continuation to this future which will send the result back
            // once its ready
            result.then(
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
    };
}}

#include <hpx/lcos/local/packaged_continuation.hpp>

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
