//  Copyright (c) 2007-2015 Hartmut Kaiser
//  Copyright (c) 2013 Agustin Berge
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_LCOS_FUTURE_MAR_06_2012_1059AM)
#define HPX_LCOS_FUTURE_MAR_06_2012_1059AM

#include <hpx/config.hpp>
#include <hpx/error_code.hpp>
#include <hpx/lcos_fwd.hpp>
#include <hpx/throw_exception.hpp>
#include <hpx/traits/acquire_shared_state.hpp>
#include <hpx/traits/is_callable.hpp>
#include <hpx/traits/is_future.hpp>
#include <hpx/traits/future_access.hpp>
#include <hpx/traits/future_traits.hpp>
#include <hpx/traits/is_launch_policy.hpp>
#include <hpx/traits/is_executor.hpp>
#include <hpx/traits/concepts.hpp>
#include <hpx/lcos/detail/future_data.hpp>
#include <hpx/util/always_void.hpp>
#include <hpx/util/bind.hpp>
#include <hpx/util/date_time_chrono.hpp>
#include <hpx/util/decay.hpp>
#include <hpx/util/invoke.hpp>
#include <hpx/util/result_of.hpp>
#include <hpx/util/void_guard.hpp>
#include <hpx/runtime/actions/continuation.hpp>
#include <hpx/runtime/launch_policy.hpp>

#include <boost/exception_ptr.hpp>
#include <boost/intrusive_ptr.hpp>
#include <boost/mpl/eval_if.hpp>
#include <boost/mpl/if.hpp>
#include <boost/type_traits/is_void.hpp>
#include <boost/utility/declval.hpp>
#include <boost/utility/enable_if.hpp>

#include <type_traits>
#include <utility>
#include <iterator>

namespace hpx { namespace lcos { namespace detail
{
    ///////////////////////////////////////////////////////////////////////////
    enum future_state
    {
        invalid = 0,
        has_value = 1,
        has_exception = 2
    };

    template <typename Archive, typename Future>
    typename boost::disable_if<
        boost::is_void<typename hpx::traits::future_traits<Future>::type>
    >::type serialize_future_load(Archive& ar, Future& f)
    {
        typedef typename hpx::traits::future_traits<Future>::type value_type;
        typedef lcos::detail::future_data<value_type> shared_state;

        int state = future_state::invalid;
        ar >> state;
        if (state == future_state::has_value)
        {
            value_type value;
            ar >> value;

            boost::intrusive_ptr<shared_state> p(new shared_state());
            p->set_value(std::move(value));

            f = hpx::traits::future_access<Future>::create(std::move(p));
        } else if (state == future_state::has_exception) {
            boost::exception_ptr exception;
            ar >> exception;

            boost::intrusive_ptr<shared_state> p(new shared_state());
            p->set_exception(exception);

            f = hpx::traits::future_access<Future>::create(std::move(p));
        } else if (state == future_state::invalid) {
            f = Future();
        } else {
            HPX_ASSERT(false);
        }
    }

    template <typename Archive, typename Future>
    typename boost::enable_if<
        boost::is_void<typename hpx::traits::future_traits<Future>::type>
    >::type serialize_future_load(Archive& ar, Future& f) //-V659
    {
        typedef lcos::detail::future_data<void> shared_state;

        int state = future_state::invalid;
        ar >> state;
        if (state == future_state::has_value)
        {
            boost::intrusive_ptr<shared_state> p(new shared_state());
            p->set_value(hpx::util::unused);

            f = hpx::traits::future_access<Future>::create(std::move(p));
        } else if (state == future_state::has_exception) {
            boost::exception_ptr exception;
            ar >> exception;

            boost::intrusive_ptr<shared_state> p(new shared_state());
            p->set_exception(exception);

            f = hpx::traits::future_access<Future>::create(std::move(p));
        } else if (state == future_state::invalid) {
            f = Future();
        } else {
            HPX_ASSERT(false);
        }
    }

    template <typename Archive, typename Future>
    typename boost::disable_if<
        boost::is_void<typename hpx::traits::future_traits<Future>::type>
    >::type serialize_future_save(Archive& ar, Future const& f)
    {
        typedef typename hpx::traits::future_traits<Future>::result_type value_type;

        if(ar.is_future_awaiting())
        {
            if(!f.is_ready())
            {
                typename hpx::traits::detail::shared_state_ptr_for<Future>::type state
                    = hpx::traits::future_access<Future>::get_shared_state(f);

                state->execute_deferred();

                ar.await_future(f);
            }
            else
            {
                if(f.is_ready() && f.has_value())
                {
                    value_type const & value =
                        *hpx::traits::future_access<Future>::
                            get_shared_state(f)->get_result();
                    ar << value;
                }
            }
            return;
        }

#if defined(HPX_DEBUG)
        if (f.valid())
        {
            HPX_ASSERT(f.is_ready());
        }
#endif

        int state = future_state::invalid;
        if (f.has_value())
        {
            state = future_state::has_value;
            if(ar.is_saving())
            {
                value_type value = const_cast<Future &>(f).get();
                ar << state << value;
            }
            else
            {
                value_type const & value =
                    *hpx::traits::future_access<Future>::
                        get_shared_state(f)->get_result();
                ar << state << value;
            }
        } else if (f.has_exception()) {
            state = future_state::has_exception;
            boost::exception_ptr exception = f.get_exception_ptr();
            ar << state << exception;
        } else {
            state = future_state::invalid;
            ar << state;
        }
    }

    template <typename Archive, typename Future>
    typename boost::enable_if<
        boost::is_void<typename hpx::traits::future_traits<Future>::type>
    >::type serialize_future_save(Archive& ar, Future const& f) //-V659
    {
        if(ar.is_future_awaiting())
        {
            if(!f.is_ready())
            {
                typename
                    hpx::traits::detail::shared_state_ptr_for<Future>::type state
                    = hpx::traits::future_access<Future>::get_shared_state(f);

                state->execute_deferred();

                ar.await_future(f);
            }
            return;
        }

#if defined(HPX_DEBUG)
        if (f.valid())
        {
            HPX_ASSERT(f.is_ready());
        }
#endif

        int state = future_state::invalid;
        if (f.has_value())
        {
            state = future_state::has_value;
        } else if (f.has_exception()) {
            state = future_state::has_exception;
            boost::exception_ptr exception = f.get_exception_ptr();
            ar << state << exception;
        } else {
            state = future_state::invalid;
            ar << state;
        }
    }

    template <typename Future>
    void serialize_future(serialization::input_archive& ar, Future& f, unsigned)
    {
        serialize_future_load(ar, f);
    }

    template <typename Future>
    void serialize_future(serialization::output_archive& ar, Future& f, unsigned)
    {
        serialize_future_save(ar, f);
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
      , typename hpx::util::always_void<
            typename hpx::util::result_of<F(Future)>::type
        >::type
    >
    {
        typedef typename hpx::util::result_of<F(Future)>::type cont_result;

        typedef typename boost::mpl::eval_if<
            hpx::traits::detail::is_unique_future<cont_result>
          , hpx::traits::future_traits<cont_result>
          , boost::mpl::identity<cont_result>
        >::type result_type;

        typedef lcos::future<result_type> type;
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename Future, typename Enable = void>
    struct future_unwrap_result
    {};

    template <template <typename> class Future, typename R>
    struct future_unwrap_result<Future<Future<R> > >
    {
        typedef R result_type;

        typedef Future<result_type> type;
    };

    template <typename R>
    struct future_unwrap_result<future<shared_future<R> > >
    {
        typedef R result_type;

        typedef future<result_type> type;
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename Iter, typename Enable = void>
    struct future_iterator_traits
    {};

    template <typename Iterator>
    struct future_iterator_traits<Iterator,
        typename hpx::util::always_void<
#if defined(HPX_MSVC) && HPX_MSVC <= 1800       // MSVC12 needs special help
            typename Iterator::iterator_category
#else
            typename std::iterator_traits<Iterator>::value_type
#endif
        >::type>
    {
        typedef
            typename std::iterator_traits<Iterator>::value_type
            type;

        typedef hpx::traits::future_traits<type> traits_type;
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename T>
    struct future_value
      : future_data_result<T>
    {
        template <typename U>
        HPX_FORCEINLINE static
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
        HPX_FORCEINLINE static
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
      : future_data_result<void>
    {
        HPX_FORCEINLINE static
        void get(hpx::util::unused_type)
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
    inline typename hpx::traits::detail::shared_state_ptr<
        typename continuation_result<ContResult>::type
    >::type
    make_continuation(Future const& future, launch policy,
        F && f);

    template <typename ContResult, typename Future, typename F>
    inline typename hpx::traits::detail::shared_state_ptr<
        typename continuation_result<ContResult>::type
    >::type
    make_continuation(Future const& future, threads::executor& sched,
        F && f);
    template <typename ContResult, typename Future, typename Executor,
        typename F>
    inline typename hpx::traits::detail::shared_state_ptr<
        typename continuation_result<ContResult>::type
    >::type
    make_continuation_exec(Future const& future, Executor& exec, F && f);

    ///////////////////////////////////////////////////////////////////////////
    template <typename Future>
    typename hpx::traits::detail::shared_state_ptr<
        typename future_unwrap_result<Future>::result_type>::type
    unwrap(Future&& future, error_code& ec = throws);

    ///////////////////////////////////////////////////////////////////////////
    class void_continuation;

    template <typename Future>
    inline typename hpx::traits::detail::shared_state_ptr<void>::type
    make_void_continuation(Future& future);

    ///////////////////////////////////////////////////////////////////////////
    template <typename Derived, typename R>
    class future_base
    {
    public:
        typedef R result_type;
        typedef future_data<R> shared_state_type;

    public:
        future_base() HPX_NOEXCEPT
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

        future_base(future_base && other) HPX_NOEXCEPT
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

        future_base& operator=(future_base && other) HPX_NOEXCEPT
        {
            if (this != &other)
            {
                shared_state_ = std::move(other.shared_state_);
                other.shared_state_ = 0;
            }
            return *this;
        }

        // Returns: true only if *this refers to a shared state.
        bool valid() const HPX_NOEXCEPT
        {
            return shared_state_ != 0;
        }

        // Returns: true if the shared state is ready, false if it isn't.
        bool is_ready() const HPX_NOEXCEPT
        {
            return shared_state_ != 0 && shared_state_->is_ready();
        }

        // Returns: true if the shared state is ready and stores a value,
        //          false if it isn't.
        bool has_value() const HPX_NOEXCEPT
        {
            return shared_state_ != 0 && shared_state_->has_value();
        }

        // Returns: true if the shared state is ready and stores an exception,
        //          false if it isn't.
        bool has_exception() const HPX_NOEXCEPT
        {
            return shared_state_ != 0 && shared_state_->has_exception();
        }

        // Effects:
        //   - Blocks until the future is ready.
        // Returns: The stored exception_ptr if has_exception(), a null
        //          pointer otherwise.
        boost::exception_ptr get_exception_ptr() const
        {
            if (!shared_state_)
            {
                HPX_THROW_EXCEPTION(no_state,
                    "future_base<R>::get_exception_ptr",
                    "this future has no valid shared state");
            }

            error_code ec(lightweight);
            this->shared_state_->get_result(ec);
            if (!ec) return boost::exception_ptr();
            return hpx::detail::access_exception(ec);
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
        typename boost::lazy_disable_if_c<
            hpx::traits::is_launch_policy<F>::value ||
            hpx::traits::is_threads_executor<F>::value ||
            hpx::traits::is_executor<F>::value
          , future_then_result<Derived, F>
        >::type
        then(F && f, error_code& ec = throws) const
        {
            return then(launch::all, std::forward<F>(f), ec);
        }

        template <typename F>
        typename future_then_result<Derived, F>::type
        then(launch policy, F && f, error_code& ec = throws) const
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
                typename hpx::util::result_of<F(Derived)>::type
                continuation_result_type;
            typedef
                typename hpx::traits::detail::shared_state_ptr<result_type>::type
                shared_state_ptr;

            shared_state_ptr p =
                detail::make_continuation<continuation_result_type>(
                    *static_cast<Derived const*>(this), policy, std::forward<F>(f));
            return hpx::traits::future_access<future<result_type> >::create(
                std::move(p));
        }

        template <typename F>
        typename future_then_result<Derived, F>::type
        then(threads::executor& sched, F && f, error_code& ec = throws) const
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
                typename hpx::util::result_of<F(Derived)>::type
                continuation_result_type;
            typedef
                typename hpx::traits::detail::shared_state_ptr<result_type>::type
                shared_state_ptr;

            shared_state_ptr p =
                detail::make_continuation<continuation_result_type>(
                    *static_cast<Derived const*>(this), sched, std::forward<F>(f));
            return hpx::traits::future_access<future<result_type> >::create(
                std::move(p));
        }

        template <typename Executor, typename F>
        typename boost::lazy_enable_if_c<
            hpx::traits::is_executor<Executor>::value
          , future_then_result<Derived, F>
        >::type
        then(Executor& exec, F && f, error_code& ec = throws) const
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
                typename hpx::util::result_of<F(Derived)>::type
                continuation_result_type;
            typedef
                typename hpx::traits::detail::shared_state_ptr<result_type>::type
                shared_state_ptr;

            shared_state_ptr p =
                detail::make_continuation_exec<continuation_result_type>(
                    *static_cast<Derived const*>(this), exec,
                    std::forward<F>(f));
            return hpx::traits::future_access<future<result_type> >::
                create(std::move(p));
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
        //          or until the absolute timeout (30.2.4) specified by
        //          abs_time has expired.
        // Returns:
        //   - future_status::deferred if the shared state contains a deferred
        //     function.
        //   - future_status::ready if the shared state is ready.
        //   - future_status::timeout if the function is returning because the
        //     absolute timeout (30.2.4) specified by abs_time has expired.
        // Throws: timeout-related exceptions (30.2.4).
        future_status
        wait_until(hpx::util::steady_time_point const& abs_time,
            error_code& ec = throws) const
        {
            if (!shared_state_)
            {
                HPX_THROWS_IF(ec, no_state,
                    "future_base<R>::wait_until",
                    "this future has no valid shared state");
                return future_status::uninitialized;
            }
            return shared_state_->wait_until(abs_time.value(), ec);
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
        future_status
        wait_for(hpx::util::steady_duration const& rel_time,
            error_code& ec = throws) const
        {
            return wait_until(rel_time.from_now(), ec);
        }

    protected:
        boost::intrusive_ptr<shared_state_type> shared_state_;
    };
}}}

namespace hpx { namespace lcos
{
    ///////////////////////////////////////////////////////////////////////////
    template <typename R>
    class future : public detail::future_base<future<R>, R>
    {
        HPX_MOVABLE_ONLY(future);

        typedef detail::future_base<future<R>, R> base_type;

    public:
        typedef R result_type;
        typedef typename base_type::shared_state_type shared_state_type;

    private:
        struct invalidate
        {
            explicit invalidate(future& f)
              : f_(f)
            {}

            ~invalidate()
            {
                f_.shared_state_.reset();
            }

            future& f_;
        };

    private:
        template <typename Future>
        friend struct hpx::traits::future_access;

        template <typename Future, typename Enable>
        friend struct hpx::traits::detail::future_access_customization_point;

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
        future() HPX_NOEXCEPT
          : base_type()
        {}

        // Effects: move constructs a future object that refers to the shared
        //          state that was originally referred to by other (if any).
        // Postconditions:
        //   - valid() returns the same value as other.valid() prior to the
        //     constructor invocation.
        //   - other.valid() == false.
        future(future && other) HPX_NOEXCEPT
          : base_type(std::move(other))
        {}

        // Effects: constructs a future object by moving the instance referred
        //          to by rhs and unwrapping the inner future.
        // Postconditions:
        //   - valid() returns the same value as other.valid() prior to the
        //     constructor invocation.
        //   - other.valid() == false.
        future(future<future> && other) HPX_NOEXCEPT
          : base_type(other.valid() ? detail::unwrap(std::move(other)) : 0)
        {}

        // Effects: constructs a future object by moving the instance referred
        //          to by rhs and unwrapping the inner future.
        // Postconditions:
        //   - valid() returns the same value as other.valid() prior to the
        //     constructor invocation.
        //   - other.valid() == false.
        future(future<shared_future<R> > && other) HPX_NOEXCEPT
          : base_type(other.valid() ? detail::unwrap(std::move(other)) : 0)
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

        // Effects:
        //   - releases any shared state (30.6.4).
        //   - move assigns the contents of other to *this.
        // Postconditions:
        //   - valid() returns the same value as other.valid() prior to the
        //     assignment.
        //   - other.valid() == false.
        future& operator=(future && other) HPX_NOEXCEPT
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
        typename hpx::traits::future_traits<future>::result_type
        get()
        {
            if (!this->shared_state_)
            {
                HPX_THROW_EXCEPTION(no_state,
                    "future<R>::get",
                    "this future has no valid shared state");
            }

            invalidate on_exit(*this);

            typedef typename shared_state_type::result_type result_type;
            result_type* result = this->shared_state_->get_result();

            // no error has been reported, return the result
            return detail::future_value<R>::get(std::move(*result));
        }

        typename hpx::traits::future_traits<future>::result_type
        get(error_code& ec)
        {
            if (!this->shared_state_)
            {
                HPX_THROWS_IF(ec, no_state,
                    "future<R>::get",
                    "this future has no valid shared state");
                return detail::future_value<R>::get_default();
            }

            invalidate on_exit(*this);

            typedef typename shared_state_type::result_type result_type;
            result_type* result = this->shared_state_->get_result(ec);
            if (ec) return detail::future_value<R>::get_default();

            // no error has been reported, return the result
            return detail::future_value<R>::get(std::move(*result));
        }
        using base_type::get_exception_ptr;

        using base_type::valid;
        using base_type::is_ready;
        using base_type::has_value;
        using base_type::has_exception;

        template <typename F>
        typename boost::lazy_disable_if_c<
            hpx::traits::is_launch_policy<F>::value ||
            hpx::traits::is_threads_executor<F>::value ||
            hpx::traits::is_executor<F>::value
          , detail::future_then_result<future, F>
        >::type
        then(F && f, error_code& ec = throws)
        {
            invalidate on_exit(*this);
            return base_type::then(std::forward<F>(f), ec);
        }

        template <typename F>
        typename detail::future_then_result<future, F>::type
        then(launch policy, F && f, error_code& ec = throws)
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

        template <typename Executor, typename F>
        typename boost::lazy_enable_if_c<
            hpx::traits::is_executor<Executor>::value
          , detail::future_then_result<future, F>
        >::type
        then(Executor& exec, F && f, error_code& ec = throws)
        {
            invalidate on_exit(*this);
            return base_type::then(exec, std::forward<F>(f), ec);
        }

        using base_type::wait;
        using base_type::wait_for;
        using base_type::wait_until;
    };

    ///////////////////////////////////////////////////////////////////////////
    namespace detail
    {
        template <typename T, typename Future>
        typename std::enable_if<
            std::is_convertible<Future, hpx::future<T> >::value,
            hpx::future<T>
        >::type make_future_helper(Future && f)
        {
            return std::move(f);
        };

        template <typename T, typename Future>
        typename std::enable_if<
            !std::is_convertible<Future, hpx::future<T> >::value,
            hpx::future<T>
        >::type make_future_helper(Future && f)
        {
            return f.then(
                [](Future && f) -> T
                {
                    return util::void_guard<T>(), f.get();
                });
        }
    }

    // Allow to convert any future<U> into any other future<R> based on an
    // existing conversion path U --> R.
    template <typename R, typename U>
    hpx::future<R>
    make_future(hpx::future<U> && f)
    {
        static_assert(
            std::is_convertible<R, U>::value || std::is_void<R>::value,
            "the argument type must be implicitly convertible to the requested "
            "result type");

        return detail::make_future_helper<R>(std::move(f));
    }

    // Allow to convert any future<U> into any other future<R> based on a given
    // conversion function: R conv(U).
    template <typename R, typename U, typename Conv>
    hpx::future<R>
    make_future(hpx::future<U> && f, Conv && conv)
    {
        static_assert(
            hpx::traits::is_callable<Conv(U), R>::value,
            "the argument type must be convertible to the requested "
            "result type by using the supplied conversion function");

        return f.then(
            [conv](hpx::future<U> && f)
            {
                return hpx::util::invoke(conv, f.get());
            });
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
        template <typename Future>
        friend struct hpx::traits::future_access;

        template <typename Future, typename Enable>
        friend struct hpx::traits::detail::future_access_customization_point;

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
        shared_future() HPX_NOEXCEPT
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
        shared_future(shared_future && other) HPX_NOEXCEPT
          : base_type(std::move(other))
        {}

        shared_future(future<R> && other) HPX_NOEXCEPT
          : base_type(hpx::traits::detail::get_shared_state(other))
        {
            other = future<R>();
        }

        // Effects: constructs a shared_future object by moving the instance
        //          referred to by rhs and unwrapping the inner future.
        // Postconditions:
        //   - valid() returns the same value as other.valid() prior to the
        //     constructor invocation.
        //   - other.valid() == false.
        shared_future(future<shared_future> && other) HPX_NOEXCEPT
          : base_type(other.valid() ? detail::unwrap(other.share()) : 0)
        {}

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
        shared_future& operator=(shared_future && other) HPX_NOEXCEPT
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
        typename hpx::traits::future_traits<shared_future>::result_type
        get() const //-V659
        {
            if (!this->shared_state_)
            {
                HPX_THROW_EXCEPTION(no_state,
                    "shared_future<R>::get",
                    "this future has no valid shared state");
            }

            typedef typename shared_state_type::result_type result_type;
            result_type* result = this->shared_state_->get_result();

            // no error has been reported, return the result
            return detail::future_value<R>::get(*result);
        }
        typename hpx::traits::future_traits<shared_future>::result_type
        get(error_code& ec) const //-V659
        {
            typedef typename shared_state_type::result_type result_type;
            if (!this->shared_state_)
            {
                HPX_THROWS_IF(ec, no_state,
                    "shared_future<R>::get",
                    "this future has no valid shared state");
                static result_type res(detail::future_value<R>::get_default());
                return res;
            }

            result_type* result = this->shared_state_->get_result(ec);
            if (ec)
            {
                static result_type res(detail::future_value<R>::get_default());
                return res;
            }

            // no error has been reported, return the result
            return detail::future_value<R>::get(*result);
        }
        using base_type::get_exception_ptr;

        using base_type::valid;
        using base_type::is_ready;
        using base_type::has_value;
        using base_type::has_exception;

        using base_type::then;

        using base_type::wait;
        using base_type::wait_for;
        using base_type::wait_until;
    };

    ///////////////////////////////////////////////////////////////////////////
    // Allow to convert any shared_future<U> into any other future<R> based on
    // an existing conversion path U --> R.
    template <typename R, typename U>
    hpx::shared_future<R>
    make_future(hpx::shared_future<U> f)
    {
        static_assert(
            std::is_convertible<R, U>::value || std::is_void<R>::value,
            "the argument type must be implicitly convertible to the requested "
            "result type");

        return detail::make_future_helper<R>(std::move(f));
    }

    // Allow to convert any future<U> into any other future<R> based on a given
    // conversion function: R conv(U).
    template <typename R, typename U, typename Conv>
    hpx::future<R>
    make_future(hpx::shared_future<U> const& f, Conv && conv)
    {
        static_assert(
            hpx::traits::is_callable<Conv(U), R>::value,
            "the argument type must be convertible to the requested "
            "result type by using the supplied conversion function");

        return f.then(
            [conv](hpx::shared_future<U> const& f)
            {
                return hpx::util::invoke(conv, f.get());
            });
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
    future<typename hpx::util::decay_unwrap<Result>::type>
    make_ready_future(Result && init)
    {
        typedef typename hpx::util::decay_unwrap<Result>::type result_type;
        typedef lcos::detail::future_data<result_type> shared_state;

        boost::intrusive_ptr<shared_state> p(new shared_state());
        p->set_value(std::forward<Result>(init));

        return hpx::traits::future_access<future<result_type> >::create(std::move(p));
    }

    // extension: create a pre-initialized future object which holds the
    // given error
    template <typename T>
    future<T> make_exceptional_future(boost::exception_ptr const& e)
    {
        typedef lcos::detail::future_data<T> shared_state;

        boost::intrusive_ptr<shared_state> p(new shared_state());
        p->set_exception(e);

        return hpx::traits::future_access<future<T> >::create(std::move(p));
    }

    template <typename T, typename E>
    future<T> make_exceptional_future(E e)
    {
        try
        {
            boost::throw_exception(e);
        } catch (...) {
            return lcos::make_exceptional_future<T>(boost::current_exception());
        }

        return future<T>();
    }

    // extension: create a pre-initialized future object which gets ready at
    // a given point in time
    template <typename Result>
    future<typename hpx::util::decay_unwrap<Result>::type>
    make_ready_future_at(hpx::util::steady_time_point const& abs_time,
        Result&& init)
    {
        typedef typename hpx::util::decay_unwrap<Result>::type result_type;
        typedef lcos::detail::timed_future_data<result_type> shared_state;

        return hpx::traits::future_access<future<result_type> >::create(
            new shared_state(abs_time.value(), std::forward<Result>(init)));
    }

    template <typename Result>
    future<typename hpx::util::decay_unwrap<Result>::type>
    make_ready_future_after(hpx::util::steady_duration const& rel_time,
        Result && init)
    {
        return make_ready_future_at(rel_time.from_now(),
            std::forward<Result>(init));
    }

    // extension: create a pre-initialized future object
    inline future<void> make_ready_future()
    {
        typedef lcos::detail::future_data<void> shared_state;

        boost::intrusive_ptr<shared_state> p(new shared_state());
        p->set_value(hpx::util::unused);

        return hpx::traits::future_access<future<void> >::create(std::move(p));
    }

    // extension: create a pre-initialized future object which gets ready at
    // a given point in time
    inline future<void> make_ready_future_at(
        hpx::util::steady_time_point const& abs_time)
    {
        typedef lcos::detail::timed_future_data<void> shared_state;

        return hpx::traits::future_access<future<void> >::create(
            new shared_state(abs_time.value(), hpx::util::unused));
    }

    inline future<void> make_ready_future_after(
        hpx::util::steady_duration const& rel_time)
    {
        return make_ready_future_at(rel_time.from_now());
    }
}}

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace actions
{
    // special handling of actions returning a future
    template <typename R>
    struct typed_continuation<lcos::future<R> > : continuation
    {
    private:
        typedef hpx::util::function<void(naming::id_type, R)> function_type;

    public:
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

        explicit typed_continuation(naming::id_type const& gid, naming::address && addr)
          : continuation(gid, std::move(addr))
        {}

        explicit typed_continuation(naming::id_type && gid, naming::address && addr)
          : continuation(std::move(gid), std::move(addr))
        {}

        template <typename F>
        explicit typed_continuation(naming::id_type const& gid,
            naming::address && addr, F && f)
          : continuation(gid, std::move(addr)), f_(std::forward<F>(f))
        {}

        template <typename F>
        explicit typed_continuation(naming::id_type && gid,
            naming::address && addr, F && f)
          : continuation(std::move(gid), std::move(addr)), f_(std::forward<F>(f))
        {}

        template <typename F,
            typename Enable
                = typename std::enable_if<
                    !std::is_same<
                        typename hpx::util::decay<F>::type, typed_continuation>::value
                    >::type
        >
        explicit typed_continuation(F && f)
          : f_(std::forward<F>(f))
        {}

        void deferred_trigger(lcos::future<R> result)
        {
            if (f_.empty()) {
                if (!this->get_id()) {
                    HPX_THROW_EXCEPTION(invalid_status,
                        "typed_continuation<lcos::future<R> >::trigger_value",
                        "attempt to trigger invalid LCO (the id is invalid)");
                    return;
                }
                hpx::set_lco_value(this->get_id(), this->get_addr(), result.get());
            }
            else {
                f_(this->get_id(), result.get());
            }
        }

        virtual void trigger_value(lcos::future<R> && result)
        {
            LLCO_(info)
                << "typed_continuation<lcos::future<R> >::trigger("
                << this->get_id() << ")";

            // if the future is ready, send the result back immediately
            if (result.is_ready()) {
                deferred_trigger(std::move(result));
                return;
            }

            // attach continuation to this future which will send the result back
            // once its ready
            result.then(
                hpx::util::bind(&typed_continuation::deferred_trigger,
                    std::move(*this),
                    hpx::util::placeholders::_1));
        }

    private:
        char const* get_continuation_name() const
        {
            return detail::get_continuation_name<typed_continuation>();
        }

        /// serialization support
        void serialize(serialization::input_archive& ar)
        {
            // serialize function
            bool have_function = false;
            ar >> have_function;
            if (have_function)
                ar >> f_;
        }
        void serialize(serialization::output_archive& ar) const
        {
            // serialize function
            bool have_function = !f_.empty();
            ar << have_function;
            if (have_function)
                ar << f_;
        }

        template <typename Archive>
        void serialize(Archive & ar, unsigned)
        {
            // serialize base class
            ar & hpx::serialization::base_object<continuation>(*this);

            serialize(ar);
        }
        HPX_SERIALIZATION_POLYMORPHIC_WITH_NAME(
            typed_continuation
          , detail::get_continuation_name<typed_continuation>()
        )

        function_type f_;
    };

    template <>
    struct typed_continuation<lcos::future<void> > : continuation
    {
    private:
        typedef hpx::util::function<void(naming::id_type)> function_type;

    public:
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

        explicit typed_continuation(naming::id_type const& gid, naming::address && addr)
          : continuation(gid, std::move(addr))
        {}

        explicit typed_continuation(naming::id_type && gid, naming::address && addr)
          : continuation(std::move(gid), std::move(addr))
        {}

        template <typename F>
        explicit typed_continuation(naming::id_type const& gid,
            naming::address && addr, F && f)
          : continuation(gid, std::move(addr)), f_(std::forward<F>(f))
        {}

        template <typename F>
        explicit typed_continuation(naming::id_type && gid,
            naming::address && addr, F && f)
          : continuation(std::move(gid), std::move(addr)), f_(std::forward<F>(f))
        {}

        template <typename F,
            typename Enable
                = typename std::enable_if<
                    !std::is_same<
                        typename hpx::util::decay<F>::type, typed_continuation>::value
                    >::type
        >
        explicit typed_continuation(F && f)
          : f_(std::forward<F>(f))
        {}

        void deferred_trigger(lcos::future<void> result)
        {
            if (f_.empty()) {
                if (!this->get_id()) {
                    HPX_THROW_EXCEPTION(invalid_status,
                        "typed_continuation<lcos::future<void> >::trigger_value",
                        "attempt to trigger invalid LCO (the id is invalid)");
                    return;
                }
                result.get();
                hpx::trigger_lco_event(this->get_id(), this->get_addr());
            }
            else {
                result.get();
                f_(this->get_id());
            }
        }

        virtual void trigger_value(lcos::future<void> && result)
        {
            LLCO_(info)
                << "typed_continuation<lcos::future<void> >::trigger("
                << this->get_id() << ")";

            // if the future is ready, send the result back immediately
            if (result.is_ready()) {
                deferred_trigger(std::move(result));
                return;
            }

            // attach continuation to this future which will send the result back
            // once its ready
            result.then(
                hpx::util::bind(&typed_continuation::deferred_trigger,
                    std::move(*this),
                    hpx::util::placeholders::_1));
        }

    private:
        char const* get_continuation_name() const
        {
            return "hpx_future_void_typed_continuation";
        }

        /// serialization support
        void serialize(serialization::input_archive& ar)
        {
            // serialize function
            bool have_function = false;
            ar >> have_function;
            if (have_function)
                ar >> f_;
        }
        void serialize(serialization::output_archive& ar)
        {
            // serialize function
            bool have_function = !f_.empty();
            ar << have_function;
            if (have_function)
                ar << f_;
        }

        template <typename Archive>
        void serialize(Archive & ar, unsigned)
        {
            // serialize base class
            ar & hpx::serialization::base_object<continuation>(*this);

            serialize(ar);
        }
        HPX_SERIALIZATION_POLYMORPHIC_WITH_NAME(
            typed_continuation
          , "hpx_future_void_typed_continuation"
        )

        function_type f_;
    };

    template <typename R>
    struct typed_continuation<lcos::shared_future<R> > : continuation
    {
    private:
        typedef hpx::util::function<void(naming::id_type, R)> function_type;

    public:
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

        explicit typed_continuation(naming::id_type const& gid, naming::address && addr)
          : continuation(gid, std::move(addr))
        {}

        explicit typed_continuation(naming::id_type && gid, naming::address && addr)
          : continuation(std::move(gid), std::move(addr))
        {}

        template <typename F>
        explicit typed_continuation(naming::id_type const& gid,
            naming::address && addr, F && f)
          : continuation(gid, std::move(addr)), f_(std::forward<F>(f))
        {}

        template <typename F>
        explicit typed_continuation(naming::id_type && gid,
            naming::address && addr, F && f)
          : continuation(std::move(gid), std::move(addr)), f_(std::forward<F>(f))
        {}

        template <typename F,
            typename Enable
                = typename std::enable_if<
                    !std::is_same<
                        typename hpx::util::decay<F>::type, typed_continuation>::value
                    >::type
        >
        explicit typed_continuation(F && f)
          : f_(std::forward<F>(f))
        {}

        void deferred_trigger(lcos::shared_future<R> result) const
        {
            if (f_.empty()) {
                if (!this->get_id()) {
                    HPX_THROW_EXCEPTION(invalid_status,
                        "typed_continuation<lcos::shared_future<R> >::trigger_value",
                        "attempt to trigger invalid LCO (the id is invalid)");
                    return;
                }
                hpx::set_lco_value(this->get_id(), this->get_addr(), result.get());
            }
            else {
                f_(this->get_id(), result.get());
            }
        }

        virtual void trigger_value(lcos::shared_future<R> && result)
        {
            LLCO_(info)
                << "typed_continuation<lcos::shared_future<R> >::trigger("
                << this->get_id() << ")";

            // if the future is ready, send the result back immediately
            if (result.is_ready()) {
                deferred_trigger(std::move(result));
                return;
            }

            // attach continuation to this future which will send the result back
            // once its ready
            result.then(
                hpx::util::bind(&typed_continuation::deferred_trigger,
                    std::move(*this),
                    hpx::util::placeholders::_1));
        }

    private:
        char const* get_continuation_name() const
        {
            return detail::get_continuation_name<typed_continuation>();
        }

        /// serialization support
        void serialize(serialization::input_archive& ar)
        {
            // serialize function
            bool have_function = false;
            ar >> have_function;
            if (have_function)
                ar >> f_;
        }
        void serialize(serialization::output_archive& ar) const
        {
            // serialize function
            bool have_function = !f_.empty();
            ar << have_function;
            if (have_function)
                ar << f_;
        }

        template <typename Archive>
        void serialize(Archive & ar, unsigned)
        {
            // serialize base class
            ar & hpx::serialization::base_object<continuation>(*this);

            serialize(ar);
        }
        HPX_SERIALIZATION_POLYMORPHIC_WITH_NAME(
            typed_continuation
          , detail::get_continuation_name<typed_continuation>()
        )

        function_type f_;
    };

    template <>
    struct typed_continuation<lcos::shared_future<void> > : continuation
    {
    private:
        typedef hpx::util::function<void(naming::id_type)> function_type;

    public:
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

        explicit typed_continuation(naming::id_type const& gid, naming::address && addr)
          : continuation(gid, std::move(addr))
        {}

        explicit typed_continuation(naming::id_type && gid, naming::address && addr)
          : continuation(std::move(gid), std::move(addr))
        {}

        template <typename F>
        explicit typed_continuation(naming::id_type const& gid,
            naming::address && addr, F && f)
          : continuation(gid, std::move(addr)), f_(std::forward<F>(f))
        {}

        template <typename F>
        explicit typed_continuation(naming::id_type && gid,
            naming::address && addr, F && f)
          : continuation(std::move(gid), std::move(addr)), f_(std::forward<F>(f))
        {}

        template <typename F,
            typename Enable
                = typename std::enable_if<
                    !std::is_same<
                        typename hpx::util::decay<F>::type, typed_continuation>::value
                    >::type
        >
        explicit typed_continuation(F && f)
          : f_(std::forward<F>(f))
        {}

        void deferred_trigger(lcos::shared_future<void> result) const
        {
            if (f_.empty()) {
                if (!this->get_id()) {
                    HPX_THROW_EXCEPTION(invalid_status,
                        "typed_continuation<lcos::shared_future<void> >::trigger_value",
                        "attempt to trigger invalid LCO (the id is invalid)");
                    return;
                }
                result.get();
                hpx::trigger_lco_event(this->get_id(), this->get_addr());
            }
            else {
                result.get();
                f_(this->get_id());
            }
        }

        virtual void trigger_value(lcos::shared_future<void> && result)
        {
            LLCO_(info)
                << "typed_continuation<lcos::shared_future<R> >::trigger("
                << this->get_id() << ")";

            // if the future is ready, send the result back immediately
            if (result.is_ready()) {
                deferred_trigger(std::move(result));
                return;
            }

            // attach continuation to this future which will send the result back
            // once its ready
            result.then(
                hpx::util::bind(&typed_continuation::deferred_trigger,
                    std::move(*this),
                    hpx::util::placeholders::_1));
        }

    private:
        char const* get_continuation_name() const
        {
            return "hpx_shared_future_void_typed_continuation";
        }

        /// serialization support
        void serialize(serialization::input_archive& ar)
        {
            // serialize function
            bool have_function = false;
            ar >> have_function;
            if (have_function)
                ar >> f_;
        }
        void serialize(serialization::output_archive& ar) const
        {
            // serialize function
            bool have_function = !f_.empty();
            ar << have_function;
            if (have_function)
                ar << f_;
        }

        template <typename Archive>
        void serialize(Archive & ar, unsigned)
        {
            // serialize base class
            ar & hpx::serialization::base_object<continuation>(*this);

            serialize(ar);
        }
        HPX_SERIALIZATION_POLYMORPHIC_WITH_NAME(
            typed_continuation
          , "hpx_shared_future_void_typed_continuation"
        )

        function_type f_;
    };
}}

namespace hpx { namespace serialization
{
    template <typename Archive, typename T>
    HPX_FORCEINLINE
    void serialize(Archive& ar, ::hpx::lcos::future<T>& f, unsigned version)
    {
        hpx::lcos::detail::serialize_future(ar, f, version);
    }

    template <typename Archive, typename T>
    HPX_FORCEINLINE
    void serialize(Archive& ar, ::hpx::lcos::shared_future<T>& f, unsigned version)
    {
        hpx::lcos::detail::serialize_future(ar, f, version);
    }
}}

#include <hpx/lcos/local/packaged_continuation.hpp>

///////////////////////////////////////////////////////////////////////////////
// hoist names into main namespace
namespace hpx
{
    using lcos::make_ready_future;
    using lcos::make_exceptional_future;
    using lcos::make_ready_future_at;
    using lcos::make_ready_future_after;

    using lcos::make_future;
}

#define HPX_MAKE_EXCEPTIONAL_FUTURE(T, errorcode, f, msg)                     \
    hpx::make_exceptional_future<T>(HPX_GET_EXCEPTION(errorcode, f, msg))     \
    /**/

#endif
