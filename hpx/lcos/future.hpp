//  Copyright (c) 2007-2018 Hartmut Kaiser
//  Copyright (c) 2013 Agustin Berge
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_LCOS_FUTURE_MAR_06_2012_1059AM)
#define HPX_LCOS_FUTURE_MAR_06_2012_1059AM

#include <hpx/config.hpp>
#include <hpx/error_code.hpp>
#include <hpx/lcos/detail/future_data.hpp>
#include <hpx/lcos/detail/future_traits.hpp>
#include <hpx/lcos_fwd.hpp>
#include <hpx/runtime/actions/continuation_fwd.hpp>
#include <hpx/runtime/serialization/detail/polymorphic_nonintrusive_factory.hpp>
#include <hpx/runtime/launch_policy.hpp>
#include <hpx/throw_exception.hpp>
#include <hpx/traits/acquire_shared_state.hpp>
#include <hpx/traits/concepts.hpp>
#include <hpx/traits/future_access.hpp>
#include <hpx/traits/future_then_result.hpp>
#include <hpx/traits/future_traits.hpp>
#include <hpx/traits/is_callable.hpp>
#if defined(HPX_HAVE_EXECUTOR_COMPATIBILITY)
#include <hpx/traits/v1/is_executor.hpp>
#endif
#include <hpx/traits/is_executor.hpp>
#include <hpx/traits/is_future.hpp>
#include <hpx/traits/is_launch_policy.hpp>
#include <hpx/util/always_void.hpp>
#include <hpx/util/assert.hpp>
#include <hpx/util/bind.hpp>
#include <hpx/util/decay.hpp>
#include <hpx/util/function.hpp>
#include <hpx/util/identity.hpp>
#include <hpx/util/invoke.hpp>
#include <hpx/util/lazy_enable_if.hpp>
#include <hpx/util/result_of.hpp>
#include <hpx/util/serialize_exception.hpp>
#include <hpx/util/steady_clock.hpp>
#include <hpx/util/thread_allocator.hpp>
#include <hpx/util/void_guard.hpp>

#if defined(HPX_HAVE_AWAIT)
    #include <hpx/lcos/detail/future_await_traits.hpp>
#endif

#include <boost/intrusive_ptr.hpp>

#include <exception>
#include <iterator>
#include <memory>
#include <type_traits>
#include <utility>

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
    void serialize_future_load(Archive& ar, Future& f, std::true_type)
    {
        typedef typename hpx::traits::future_traits<Future>::type value_type;
        typedef lcos::detail::future_data<value_type> shared_state;
        typedef typename shared_state::init_no_addref init_no_addref;

        value_type value;
        ar >> value;

        boost::intrusive_ptr<shared_state> p(
            new shared_state(init_no_addref{}, std::move(value)), false);

        f = hpx::traits::future_access<Future>::create(std::move(p));
    }

    template <typename Archive, typename Future>
    void serialize_future_load(Archive& ar, Future& f, std::false_type)
    {
        typedef typename hpx::traits::future_traits<Future>::type value_type;
        typedef lcos::detail::future_data<value_type> shared_state;
        typedef typename shared_state::init_no_addref init_no_addref;

        std::unique_ptr<value_type> value(
            serialization::detail::constructor_selector<value_type>::create(ar));

        boost::intrusive_ptr<shared_state> p(
            new shared_state(init_no_addref{}, std::move(*value)), false);

        f = hpx::traits::future_access<Future>::create(std::move(p));
    }

    template <typename Archive, typename Future>
    typename std::enable_if<
        !std::is_void<typename hpx::traits::future_traits<Future>::type>::value
    >::type serialize_future_load(Archive& ar, Future& f)
    {
        typedef typename hpx::traits::future_traits<Future>::type value_type;
        typedef lcos::detail::future_data<value_type> shared_state;
        typedef typename shared_state::init_no_addref init_no_addref;

        int state = future_state::invalid;
        ar >> state;
        if (state == future_state::has_value)
        {
            serialize_future_load(
                ar, f, std::is_default_constructible<value_type>());

        } else if (state == future_state::has_exception) {
            std::exception_ptr exception;
            ar >> exception;

            boost::intrusive_ptr<shared_state> p(
                new shared_state(init_no_addref{}, std::move(exception)), false);

            f = hpx::traits::future_access<Future>::create(std::move(p));
        } else if (state == future_state::invalid) {
            f = Future();
        } else {
            HPX_THROW_EXCEPTION(invalid_status,
                "serialize_future_load",
                "attempting to deserialize a future with an unknown state");
        }
    }

    template <typename Archive, typename Future>
    typename std::enable_if<
        std::is_void<typename hpx::traits::future_traits<Future>::type>::value
    >::type serialize_future_load(Archive& ar, Future& f) //-V659
    {
        typedef lcos::detail::future_data<void> shared_state;
        typedef typename shared_state::init_no_addref init_no_addref;

        int state = future_state::invalid;
        ar >> state;
        if (state == future_state::has_value)
        {
            boost::intrusive_ptr<shared_state> p(
                new shared_state(init_no_addref{}, hpx::util::unused), false);

            f = hpx::traits::future_access<Future>::create(std::move(p));
        } else if (state == future_state::has_exception) {
            std::exception_ptr exception;
            ar >> exception;

            boost::intrusive_ptr<shared_state> p(
                new shared_state(init_no_addref{}, std::move(exception)), false);

            f = hpx::traits::future_access<Future>::create(std::move(p));
        } else if (state == future_state::invalid) {
            f = Future();
        } else {
            HPX_THROW_EXCEPTION(invalid_status,
                "serialize_future_load",
                "attempting to deserialize a future with an unknown state");
        }
    }

    template <typename Archive, typename T>
    void serialize_future_save(Archive& ar, T const& val, std::false_type)
    {
        using serialization::detail::save_construct_data;
        save_construct_data(ar, &val, 0);
        ar << val;
    }

    template <typename Archive, typename T>
    void serialize_future_save(Archive& ar, T const& val, std::true_type)
    {
        ar << val;
    }

    template <typename Archive, typename Future>
    typename std::enable_if<
        !std::is_void<typename hpx::traits::future_traits<Future>::type>::value
    >::type serialize_future_save(Archive& ar, Future const& f)
    {
        typedef typename hpx::traits::future_traits<Future>::result_type value_type;

        int state = future_state::invalid;
        if (f.valid() && !f.is_ready())
        {
            if (ar.is_preprocessing())
            {
                typename hpx::traits::detail::shared_state_ptr_for<Future>::type state
                    = hpx::traits::future_access<Future>::get_shared_state(f);

                state->execute_deferred();

                ar.await_future(f);
            }
            else
            {
                HPX_THROW_EXCEPTION(invalid_status,
                    "serialize_future_save",
                    "future must be ready in order for it to be serialized");
            }
            return;
        }

        if (f.has_value())
        {
            state = future_state::has_value;
            value_type const & value =
                *hpx::traits::future_access<Future>::
                    get_shared_state(f)->get_result();
            ar << state;

            serialize_future_save(
                ar, value, std::is_default_constructible<value_type>());

        } else if (f.has_exception()) {
            state = future_state::has_exception;
            std::exception_ptr exception = f.get_exception_ptr();
            ar << state << exception;
        } else {
            state = future_state::invalid;
            ar << state;
        }
    }

    template <typename Archive, typename Future>
    typename std::enable_if<
        std::is_void<typename hpx::traits::future_traits<Future>::type>::value
    >::type serialize_future_save(Archive& ar, Future const& f) //-V659
    {
        int state = future_state::invalid;
        if (f.valid() && !f.is_ready())
        {
            if (ar.is_preprocessing())
            {
                typename
                    hpx::traits::detail::shared_state_ptr_for<Future>::type state
                    = hpx::traits::future_access<Future>::get_shared_state(f);

                state->execute_deferred();

                ar.await_future(f);
            }
            else
            {
                HPX_THROW_EXCEPTION(invalid_status,
                    "serialize_future_save",
                    "future must be ready in order for it to be serialized");
            }
            return;
        }

        if (f.has_value())
        {
            state = future_state::has_value;
            ar << state;
        }
        else if (f.has_exception())
        {
            state = future_state::has_exception;
            std::exception_ptr exception = f.get_exception_ptr();
            ar << state << exception;
        }
        else
        {
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
    template <typename R>
    struct future_get_result
    {
        template <typename SharedState>
        HPX_FORCEINLINE static R*
        call(SharedState const& state, error_code& ec = throws)
        {
            return state->get_result(ec);
        }
    };

    template <>
    struct future_get_result<util::unused_type>
    {
        template <typename SharedState>
        HPX_FORCEINLINE static util::unused_type*
        call(SharedState const& state, error_code& ec = throws)
        {
            return state->get_result_void(ec);
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename Future, typename F, typename ContResult>
    class continuation;

    template <typename ContResult>
    struct continuation_result;

    template <typename ContResult, typename Future, typename Policy, typename F>
    inline typename hpx::traits::detail::shared_state_ptr<
        typename continuation_result<ContResult>::type
    >::type
    make_continuation(Future const& future, Policy && policy, F && f);

#if defined(HPX_HAVE_EXECUTOR_COMPATIBILITY)
    template <typename ContResult, typename Future, typename Executor,
        typename F>
    inline typename hpx::traits::detail::shared_state_ptr<
        typename continuation_result<ContResult>::type
    >::type
    make_continuation_exec_v1(Future const& future, Executor& exec, F && f);
#endif

    // create non-unwrapping continuations
    template <typename ContResult, typename Future, typename Executor,
        typename F>
    inline typename traits::detail::shared_state_ptr<ContResult>::type
    make_continuation_exec(Future const& future, Executor && exec, F && f);

    template <typename Executor, typename Future, typename F>
    inline typename hpx::traits::future_then_executor_result<
        Executor, typename std::decay<Future>::type, F
    >::type
    then_execute_helper(Executor &&, F &&, Future &&);

    ///////////////////////////////////////////////////////////////////////////
    template <typename Future, typename FD, typename Enable = void>
    struct future_then_dispatch;

    // launch
    template <typename Future, typename Policy>
    struct future_then_dispatch<Future, Policy,
        typename std::enable_if<
            traits::is_launch_policy<Policy>::value
        >::type>
    {
        template <typename Policy_, typename F>
        HPX_FORCEINLINE
        static typename hpx::traits::future_then_result<Future, F>::type
        call(Future && fut, Policy_ && policy, F && f)
        {
            using result_type =
                typename hpx::traits::future_then_result<Future, F>::result_type;
            using continuation_result_type =
                typename hpx::util::invoke_result<F, Future>::type;

            typename hpx::traits::detail::shared_state_ptr<result_type>::type p =
                detail::make_continuation<continuation_result_type>(
                    std::move(fut), std::forward<Policy_>(policy),
                    std::forward<F>(f));
            return hpx::traits::future_access<future<result_type> >::create(
                std::move(p));
        }
    };

#if defined(HPX_HAVE_EXECUTOR_COMPATIBILITY)
    // parallel executors v1
    template <typename Future, typename Executor>
    struct future_then_dispatch<Future, Executor,
        typename std::enable_if<
            traits::is_executor<Executor>::value
        >::type>
    {
        template <typename Executor_, typename F>
        HPX_FORCEINLINE
        static typename hpx::traits::future_then_result<Future, F>::type
        call(Future && fut, Executor_ && exec, F && f)
        {
            using result_type =
                typename hpx::traits::future_then_result<Future, F>::result_type;
            using continuation_result_type =
                typename hpx::util::invoke_result<F, Future>::type;

            typename hpx::traits::detail::shared_state_ptr<result_type>::type p =
                detail::make_continuation_exec_v1<continuation_result_type>(
                    std::move(fut), std::forward<Executor_>(exec),
                    std::forward<F>(f));
            return hpx::traits::future_access<future<result_type> >::create(
                std::move(p));
        }
    };
#endif

    // The overload for future::then taking an executor simply forwards to the
    // corresponding executor customization point.
    //
    // parallel executors v2
    // threads::executor
    template <typename Future, typename Executor>
    struct future_then_dispatch<Future, Executor,
        typename std::enable_if<
            traits::is_one_way_executor<Executor>::value ||
            traits::is_two_way_executor<Executor>::value ||
            traits::is_threads_executor<Executor>::value
        >::type>
    {
        template <typename Executor_, typename F>
        HPX_FORCEINLINE
        static typename hpx::traits::future_then_executor_result<
            Executor_, Future, F>::type
        call(Future && fut, Executor_ && exec, F && f)
        {
            // simply forward this to executor
            return detail::then_execute_helper(std::forward<Executor_>(exec),
                std::forward<F>(f), std::move(fut));
        }
    };

    // plain function, or function object
    template <typename Future, typename FD>
    struct future_then_dispatch<Future, FD, typename std::enable_if<
        !traits::is_launch_policy<FD>::value &&
#if defined(HPX_HAVE_EXECUTOR_COMPATIBILITY)
        !traits::is_executor<FD>::value &&
#endif
        !(
            traits::is_one_way_executor<FD>::value ||
            traits::is_two_way_executor<FD>::value ||
            traits::is_threads_executor<FD>::value)
        >::type>
    {
        template <typename F>
        HPX_FORCEINLINE static auto
        call(Future && fut, F && f)
        ->  decltype(future_then_dispatch<Future, launch>::call(
                std::move(fut), launch::all, std::forward<F>(f)))
        {
            return future_then_dispatch<Future, launch>::call(
                std::move(fut), launch::all, std::forward<F>(f));
        }
    };

    ///////////////////////////////////////////////////////////////////////////
//     template <typename R>
//     typename hpx::traits::detail::shared_state_ptr<
//         typename future_unwrap_result<future<R>>::result_type>::type
//     unwrap(future<R> && future, error_code& ec = throws);

    template <typename Future>
    typename hpx::traits::detail::shared_state_ptr<
        typename future_unwrap_result<Future>::result_type>::type
    unwrap(Future && future, error_code& ec = throws);

    ///////////////////////////////////////////////////////////////////////////
    template <typename Future>
    inline typename hpx::traits::detail::shared_state_ptr<void>::type
    downcast_to_void(Future& future, bool addref);

    ///////////////////////////////////////////////////////////////////////////
    template <typename Derived, typename R>
    class future_base
    {
    public:
        typedef R result_type;
        typedef future_data_base<
                typename traits::detail::shared_state_ptr_result<R>::type
            > shared_state_type;

    private:
        template <typename F>
        struct future_then_dispatch
          : detail::future_then_dispatch<Derived, F>
        {};

    public:
        future_base() noexcept
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

        future_base(future_base && other) noexcept
          : shared_state_(std::move(other.shared_state_))
        {
            other.shared_state_ = nullptr;
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

        future_base& operator=(future_base && other) noexcept
        {
            if (this != &other)
            {
                shared_state_ = std::move(other.shared_state_);
                other.shared_state_ = nullptr;
            }
            return *this;
        }

        // Returns: true only if *this refers to a shared state.
        bool valid() const noexcept
        {
            return shared_state_ != nullptr;
        }

        // Returns: true if the shared state is ready, false if it isn't.
        bool is_ready() const noexcept
        {
            return shared_state_ != nullptr && shared_state_->is_ready();
        }

        // Returns: true if the shared state is ready and stores a value,
        //          false if it isn't.
        bool has_value() const noexcept
        {
            return shared_state_ != nullptr && shared_state_->has_value();
        }

        // Returns: true if the shared state is ready and stores an exception,
        //          false if it isn't.
        bool has_exception() const noexcept
        {
            return shared_state_ != nullptr && shared_state_->has_exception();
        }

        // Effects:
        //   - Blocks until the future is ready.
        // Returns: The stored exception_ptr if has_exception(), a null
        //          pointer otherwise.
        std::exception_ptr get_exception_ptr() const
        {
            if (!shared_state_)
            {
                HPX_THROW_EXCEPTION(no_state,
                    "future_base<R>::get_exception_ptr",
                    "this future has no valid shared state");
            }

            typedef typename shared_state_type::result_type result_type;

            error_code ec(lightweight);
            detail::future_get_result<result_type>::call(this->shared_state_, ec);
            if (!ec)
            {
                HPX_ASSERT(!has_exception());
                return std::exception_ptr();
            }
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
        static auto then(Derived && fut, F && f, error_code& ec = throws)
        ->  decltype(future_then_dispatch<typename std::decay<F>::type>::call(
                std::move(fut), std::forward<F>(f)))
        {
            using result_type =
                decltype(future_then_dispatch<typename std::decay<F>::type>::call(
                    std::move(fut), std::forward<F>(f)));

            if (!fut.shared_state_)
            {
                HPX_THROWS_IF(ec, no_state,
                    "future_base<R>::then",
                    "this future has no valid shared state");
                return result_type();
            }

            return future_then_dispatch<typename std::decay<F>::type>::call(
                std::move(fut), std::forward<F>(f));
        }

        template <typename F, typename T0>
        static auto then(Derived && fut, T0 && t0, F && f, error_code& ec = throws)
        ->  decltype(future_then_dispatch<typename std::decay<T0>::type>::call(
                std::move(fut), std::forward<T0>(t0), std::forward<F>(f)))
        {
            using result_type =
                decltype(future_then_dispatch<typename std::decay<T0>::type>::call(
                    std::move(fut), std::forward<T0>(t0), std::forward<F>(f)));

            if (!fut.shared_state_)
            {
                HPX_THROWS_IF(ec, no_state,
                    "future_base<R>::then",
                    "this future has no valid shared state");
                return result_type();
            }

            return future_then_dispatch<typename std::decay<T0>::type>::call(
                std::move(fut), std::forward<T0>(t0), std::forward<F>(f));
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

#if defined(HPX_HAVE_AWAIT)
        bool await_ready() const
        {
            return detail::await_ready(*static_cast<Derived const*>(this));
        }

        template <typename Promise>
        void await_suspend(std::experimental::coroutine_handle<Promise> rh)
        {
            detail::await_suspend(*static_cast<Derived*>(this), rh);
        }

        auto await_resume()
        ->  decltype(detail::await_resume(std::declval<Derived>()))
        {
            return detail::await_resume(*static_cast<Derived*>(this));
        }
#endif

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
        future() noexcept
          : base_type()
        {}

        // Effects: move constructs a future object that refers to the shared
        //          state that was originally referred to by other (if any).
        // Postconditions:
        //   - valid() returns the same value as other.valid() prior to the
        //     constructor invocation.
        //   - other.valid() == false.
        future(future && other) noexcept
          : base_type(std::move(other))
        {}

        // Effects: constructs a future object by moving the instance referred
        //          to by rhs and unwrapping the inner future.
        // Postconditions:
        //   - valid() returns the same value as other.valid() prior to the
        //     constructor invocation.
        //   - other.valid() == false.
        future(future<future> && other) noexcept
          : base_type(other.valid() ? detail::unwrap(std::move(other)) : nullptr)
        {}

        // Effects: constructs a future object by moving the instance referred
        //          to by rhs and unwrapping the inner future.
        // Postconditions:
        //   - valid() returns the same value as other.valid() prior to the
        //     constructor invocation.
        //   - other.valid() == false.
        future(future<shared_future<R> > && other) noexcept
          : base_type(other.valid() ? detail::unwrap(std::move(other)) : nullptr)
        {}

        // Effects: constructs a future<void> object that will be ready when
        //          the given future is ready
        // Postconditions:
        //   - valid() returns the same value as other.valid() prior to the
        //     constructor invocation.
        //   - other.valid() == false.
        template <typename T>
        future(future<T>&& other,
            typename std::enable_if<
                std::is_void<R>::value && !traits::is_future<T>::value, T
            >::type* = nullptr
        ) : base_type(other.valid() ?
                detail::downcast_to_void(other, false) : nullptr)
        {
            traits::future_access<future<T> >::
                detach_shared_state(std::move(other));
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
        future& operator=(future && other) noexcept
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
            result_type* result = detail::future_get_result<result_type>::call(
                this->shared_state_);

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
            result_type* result = detail::future_get_result<result_type>::call(
                this->shared_state_, ec);
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
        typename util::lazy_enable_if<
            !hpx::traits::is_launch_policy<
                typename std::decay<F>::type>::value &&
#if defined(HPX_HAVE_EXECUTOR_COMPATIBILITY)
            !hpx::traits::is_executor<
                typename std::decay<F>::type>::value &&
#endif
            !hpx::traits::is_one_way_executor<
                typename std::decay<F>::type>::value &&
            !hpx::traits::is_two_way_executor<
                typename std::decay<F>::type>::value &&
            !hpx::traits::is_threads_executor<
                typename std::decay<F>::type>::value
          , hpx::traits::future_then_result<future, F>
        >::type
        then(F && f, error_code& ec = throws)
        {
            invalidate on_exit(*this);
            return base_type::then(std::move(*this),
                std::forward<F>(f), ec);
        }

        template <typename Policy, typename F>
        typename util::lazy_enable_if<
            hpx::traits::is_launch_policy<
                typename std::decay<Policy>::type
            >::value,
            hpx::traits::future_then_result<future, F>
        >::type
        then(Policy && policy, F && f, error_code& ec = throws)
        {
            invalidate on_exit(*this);
            return base_type::then(std::move(*this),
                std::forward<Policy>(policy), std::forward<F>(f), ec);
        }

#if defined(HPX_HAVE_EXECUTOR_COMPATIBILITY)
        template <typename Executor, typename F>
        HPX_DEPRECATED(HPX_DEPRECATED_MSG)
        typename util::lazy_enable_if<
            hpx::traits::is_executor<Executor>::value
          , hpx::traits::future_then_result<future, F>
        >::type
        then(Executor && exec, F && f, error_code& ec = throws)
        {
            invalidate on_exit(*this);
            return base_type::then(std::move(*this),
                exec, std::forward<F>(f), ec);
        }
#endif

        template <typename Executor, typename F>
        typename util::lazy_enable_if<
            hpx::traits::is_one_way_executor<
                typename std::decay<Executor>::type>::value ||
            hpx::traits::is_two_way_executor<
                typename std::decay<Executor>::type>::value ||
            hpx::traits::is_threads_executor<
                typename std::decay<Executor>::type>::value
          , hpx::traits::future_then_executor_result<Executor, future, F>
        >::type
        then(Executor && exec, F && f, error_code& ec = throws)
        {
            invalidate on_exit(*this);
            return base_type::then(std::move(*this),
                exec, std::forward<F>(f), ec);
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
        >::type
        make_future_helper(Future && f)
        {
            return std::forward<Future>(f);
        }

        template <typename T, typename Future>
        typename std::enable_if<
           !std::is_convertible<Future, hpx::future<T> >::value,
            hpx::future<T>
        >::type
        make_future_helper(Future && f) //-V659
        {
            return f.then(
                hpx::launch::sync,
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

    namespace detail
    {
        template <typename T, typename Future, typename Conv>
        typename std::enable_if<
            std::is_convertible<Future, hpx::future<T> >::value,
            hpx::future<T>
        >::type
        convert_future_helper(Future && f, Conv && conv)
        {
            return std::forward<Future>(f);
        }

        template <typename T, typename Future, typename Conv>
        typename std::enable_if<
           !std::is_convertible<Future, hpx::future<T> >::value,
            hpx::future<T>
        >::type
        convert_future_helper(Future && f, Conv && conv) //-V659
        {
            return f.then(
                hpx::launch::sync,
                [HPX_CAPTURE_FORWARD(conv)](Future && f) -> T
                {
                    return hpx::util::invoke(conv, f.get());
                });
        }
    }

    // Allow to convert any future<U> into any other future<R> based on a given
    // conversion function: R conv(U).
    template <typename R, typename U, typename Conv>
    hpx::future<R>
    make_future(hpx::future<U> && f, Conv && conv)
    {
        return detail::convert_future_helper<R>(
            std::move(f), std::forward<Conv>(conv));
    }
}}

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
        shared_future() noexcept
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
        shared_future(shared_future && other) noexcept
          : base_type(std::move(other))
        {}

        shared_future(future<R> && other) noexcept
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
        shared_future(future<shared_future> && other) noexcept
          : base_type(other.valid() ? detail::unwrap(other.share()) : nullptr)
        {}

        // Effects: constructs a future<void> object that will be ready when
        //          the given future is ready
        // Postconditions:
        //   - valid() returns the same value as other.valid() prior to the
        //     constructor invocation.
        template <typename T>
        shared_future(shared_future<T> const& other,
            typename std::enable_if<
                std::is_void<R>::value && !traits::is_future<T>::value, T
            >::type* = nullptr
        ) : base_type(other.valid() ?
                detail::downcast_to_void(other, true) : nullptr)
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
        shared_future& operator=(shared_future && other) noexcept
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
            result_type* result = detail::future_get_result<result_type>::call(
                this->shared_state_);

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

            result_type* result = detail::future_get_result<result_type>::call(
                this->shared_state_, ec);
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

        template <typename F>
        typename util::lazy_enable_if<
            !hpx::traits::is_launch_policy<
                typename std::decay<F>::type>::value &&
            !hpx::traits::is_threads_executor<
                typename std::decay<F>::type>::value &&
#if defined(HPX_HAVE_EXECUTOR_COMPATIBILITY)
            !hpx::traits::is_executor<
                typename std::decay<F>::type>::value &&
#endif
            !hpx::traits::is_one_way_executor<
                typename std::decay<F>::type>::value &&
            !hpx::traits::is_two_way_executor<
                typename std::decay<F>::type>::value
          , hpx::traits::future_then_result<shared_future, F>
        >::type
        then(F && f, error_code& ec = throws) const
        {
            return base_type::then(shared_future(*this),
                std::forward<F>(f), ec);
        }

        template <typename Policy, typename F>
        typename util::lazy_enable_if<
            hpx::traits::is_launch_policy<
                typename std::decay<Policy>::type
            >::value,
            hpx::traits::future_then_result<shared_future, F>
        >::type
        then(Policy && policy, F && f, error_code& ec = throws) const
        {
            return base_type::then(shared_future(*this),
                std::forward<Policy>(policy), std::forward<F>(f), ec);
        }

        template <typename F>
        typename hpx::traits::future_then_result<shared_future, F>::type
        then(threads::executor& sched, F && f, error_code& ec = throws) const
        {
            return base_type::then(shared_future(*this),
                sched, std::forward<F>(f), ec);
        }

#if defined(HPX_HAVE_EXECUTOR_COMPATIBILITY)
        template <typename Executor, typename F>
        HPX_DEPRECATED(HPX_DEPRECATED_MSG)
        typename util::lazy_enable_if<
            hpx::traits::is_executor<Executor>::value
          , hpx::traits::future_then_result<shared_future, F>
        >::type
        then(Executor && exec, F && f, error_code& ec = throws) const
        {
            return base_type::then(shared_future(*this),
                exec, std::forward<F>(f), ec);
        }
#endif

        template <typename Executor, typename F>
        typename util::lazy_enable_if<
            hpx::traits::is_one_way_executor<
                typename std::decay<Executor>::type>::value ||
            hpx::traits::is_two_way_executor<
                typename std::decay<Executor>::type>::value
          , hpx::traits::future_then_executor_result<Executor, shared_future, F>
        >::type
        then(Executor && exec, F && f, error_code& ec = throws) const
        {
            return base_type::then(shared_future(*this),
                exec, std::forward<F>(f), ec);
        }

        using base_type::wait;
        using base_type::wait_for;
        using base_type::wait_until;
    };

    ///////////////////////////////////////////////////////////////////////////
    // Allow to convert any shared_future<U> into any other future<R> based on
    // an existing conversion path U --> R.
    template <typename R, typename U>
    hpx::future<R>
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
            hpx::traits::is_invocable_r<R, Conv, U>::value,
            "the argument type must be convertible to the requested "
            "result type by using the supplied conversion function");

        return f.then(
                hpx::launch::sync,
            [HPX_CAPTURE_FORWARD(conv)](hpx::shared_future<U> const& f)
            {
                return hpx::util::invoke(conv, f.get());
            });
    }

    ///////////////////////////////////////////////////////////////////////////
    // Convert any type of future<T> or shared_future<T> into a corresponding
    // shared_future<T>.
    template <typename R>
    hpx::shared_future<R>
    make_shared_future(hpx::future<R> && f)
    {
        return f.share();
    }

    template <typename R>
    hpx::shared_future<R>&
    make_shared_future(hpx::shared_future<R>& f)
    {
        return f;
    }

    template <typename R>
    hpx::shared_future<R> &&
    make_shared_future(hpx::shared_future<R> && f)
    {
        return f;
    }

    template <typename R>
    hpx::shared_future<R> const&
    make_shared_future(hpx::shared_future<R> const& f)
    {
        return f;
    }
}}

namespace hpx { namespace lcos
{
    ///////////////////////////////////////////////////////////////////////////
    // extension: create a pre-initialized future object
    template <typename T>
    typename std::enable_if<
       !std::is_void<T>::value,
        future<typename hpx::util::decay_unwrap<T>::type>
    >::type
    make_ready_future(T&& init)
    {
        using result_type = typename hpx::util::decay_unwrap<T>::type;

        using base_allocator =
            util::thread_allocator<detail::future_data_refcnt_base>;
        using shared_state = typename traits::detail::shared_state_allocator<
                lcos::detail::future_data<result_type>, base_allocator
            >::type;

        using allocator = typename std::allocator_traits<base_allocator>::
            template rebind_alloc<shared_state>;
        using traits = std::allocator_traits<allocator>;

        using init_no_addref = typename shared_state::init_no_addref;

        using unique_ptr = std::unique_ptr<shared_state,
            util::allocator_deleter<allocator>>;

        allocator alloc{};
        unique_ptr p(traits::allocate(alloc, 1),
            util::allocator_deleter<allocator>{alloc});
        traits::construct(
            alloc, p.get(), init_no_addref{}, alloc, std::forward<T>(init));

        return hpx::traits::future_access<future<result_type>>::create(
            p.release(), false);
    }

    // Extension (see wg21.link/P0319)
    template <typename T>
    typename std::enable_if<
        std::is_constructible<T>::value && !std::is_void<T>::value,
        future<typename hpx::util::decay_unwrap<T>::type>
    >::type
    make_ready_future()
    {
        using result_type = typename hpx::util::decay_unwrap<T>::type;

        using base_allocator =
            util::thread_allocator<detail::future_data_refcnt_base>;
        using shared_state = typename traits::detail::shared_state_allocator<
                lcos::detail::future_data<result_type>, base_allocator
            >::type;

        using allocator = typename std::allocator_traits<base_allocator>::
            template rebind_alloc<shared_state>;
        using traits = std::allocator_traits<allocator>;

        using init_no_addref = typename shared_state::init_no_addref;
        using default_construct = typename shared_state::default_construct;

        using unique_ptr = std::unique_ptr<shared_state,
            util::allocator_deleter<allocator>>;

        allocator alloc{};
        unique_ptr p(traits::allocate(alloc, 1),
            util::allocator_deleter<allocator>{alloc});
        traits::construct(
            alloc, p.get(), init_no_addref{}, default_construct{}, alloc);

        return hpx::traits::future_access<future<result_type>>::create(
            p.release(), false);
    }

    // Extension (see wg21.link/P0319)
    template <typename T, typename T1, typename ... Ts>
    typename std::enable_if<
        std::is_constructible<T, T1&&, Ts&&...>::value,
        future<typename hpx::util::decay_unwrap<T>::type>
    >::type
    make_ready_future(T1&& t1, Ts&&... ts)
    {
        using result_type = T;

        using base_allocator =
            util::thread_allocator<detail::future_data_refcnt_base>;
        using shared_state = typename traits::detail::shared_state_allocator<
                lcos::detail::future_data<result_type>, base_allocator
            >::type;

        using allocator = typename std::allocator_traits<base_allocator>::
            template rebind_alloc<shared_state>;
        using traits = std::allocator_traits<allocator>;

        using init_no_addref = typename shared_state::init_no_addref;

        using unique_ptr = std::unique_ptr<shared_state,
            util::allocator_deleter<allocator>>;

        allocator alloc{};
        unique_ptr p(traits::allocate(alloc, 1),
            util::allocator_deleter<allocator>{alloc});
        traits::construct(alloc, p.get(), init_no_addref{}, alloc,
            std::forward<T1>(t1), std::forward<Ts>(ts)...);

        return hpx::traits::future_access<future<result_type>>::create(
            p.release(), false);
    }

    ///////////////////////////////////////////////////////////////////////////
    // extension: create a pre-initialized future object which holds the
    // given error
    template <typename T>
    future<T> make_exceptional_future(std::exception_ptr const& e)
    {
        typedef lcos::detail::future_data<T> shared_state;
        typedef typename shared_state::init_no_addref init_no_addref;

        boost::intrusive_ptr<shared_state> p(
            new shared_state(init_no_addref{}, e), false);

        return hpx::traits::future_access<future<T> >::create(std::move(p));
    }

    template <typename T, typename E>
    future<T> make_exceptional_future(E e)
    {
        try
        {
            throw e;
        } catch (...) {
            return lcos::make_exceptional_future<T>(std::current_exception());
        }

        return future<T>();
    }

    ///////////////////////////////////////////////////////////////////////////
    // extension: create a pre-initialized future object which gets ready at
    // a given point in time
    template <typename T>
    typename std::enable_if<
       !std::is_void<T>::value,
        future<typename hpx::util::decay_unwrap<T>::type>
    >::type
    make_ready_future_at(hpx::util::steady_time_point const& abs_time,
        T&& init)
    {
        typedef typename hpx::util::decay_unwrap<T>::type result_type;
        typedef lcos::detail::timed_future_data<result_type> shared_state;

        boost::intrusive_ptr<shared_state> p(
            new shared_state(abs_time.value(), std::forward<T>(init)));

        return hpx::traits::future_access<future<result_type> >::create(
            std::move(p));
    }

    template <typename T>
    typename std::enable_if<
       !std::is_void<T>::value,
        future<typename hpx::util::decay_unwrap<T>::type>
    >::type
    make_ready_future_after(hpx::util::steady_duration const& rel_time,
        T&& init)
    {
        return make_ready_future_at(rel_time.from_now(),
            std::forward<T>(init));
    }

    ///////////////////////////////////////////////////////////////////////////
    // extension: create a pre-initialized future object
    inline future<void> make_ready_future()
    {
        using base_allocator =
            util::thread_allocator<detail::future_data_refcnt_base>;
        using shared_state = typename traits::detail::shared_state_allocator<
                lcos::detail::future_data<void>, base_allocator
            >::type;

        using allocator = typename std::allocator_traits<base_allocator>::
            template rebind_alloc<shared_state>;
        using traits = std::allocator_traits<allocator>;

        using init_no_addref = typename shared_state::init_no_addref;

        using unique_ptr = std::unique_ptr<shared_state,
            util::allocator_deleter<allocator>>;

        allocator alloc{};
        unique_ptr p(traits::allocate(alloc, 1),
            util::allocator_deleter<allocator>{alloc});
        traits::construct(
            alloc, p.get(), init_no_addref{}, alloc, util::unused);

        return hpx::traits::future_access<future<void>>::create(
            p.release(), false);
    }

    // Extension (see wg21.link/P0319)
    template <typename T>
    typename std::enable_if<std::is_void<T>::value, future<void> >::type
    make_ready_future()
    {
        return make_ready_future();
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

    template <typename T>
    typename std::enable_if<std::is_void<T>::value, future<void> >::type
    make_ready_future_at(hpx::util::steady_time_point const& abs_time)
    {
        return make_ready_future_at(abs_time);
    }

    ///////////////////////////////////////////////////////////////////////////
    inline future<void> make_ready_future_after(
        hpx::util::steady_duration const& rel_time)
    {
        return make_ready_future_at(rel_time.from_now());
    }

    template <typename T>
    typename std::enable_if<std::is_void<T>::value, future<void> >::type
    make_ready_future_after(hpx::util::steady_duration const& rel_time)
    {
        return make_ready_future_at(rel_time.from_now());
    }
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

///////////////////////////////////////////////////////////////////////////////
// hoist names into main namespace
namespace hpx
{
    using lcos::make_ready_future;
    using lcos::make_exceptional_future;
    using lcos::make_ready_future_at;
    using lcos::make_ready_future_after;

    using lcos::make_future;

    using lcos::make_shared_future;
}

#include <hpx/lcos/local/packaged_continuation.hpp>

#define HPX_MAKE_EXCEPTIONAL_FUTURE(T, errorcode, f, msg)                     \
    hpx::make_exceptional_future<T>(HPX_GET_EXCEPTION(errorcode, f, msg))     \
    /**/

#endif
