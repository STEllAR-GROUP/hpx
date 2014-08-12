//  Copyright (c) 2007-2013 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !BOOST_PP_IS_ITERATING

#if !defined(HPX_ASYNC_APR_16_20012_0225PM)
#define HPX_ASYNC_APR_16_20012_0225PM

#include <hpx/hpx_fwd.hpp>
#include <hpx/lcos/async.hpp>
#include <hpx/lcos/async_colocated.hpp>
#include <hpx/lcos/async_continue.hpp>
#include <hpx/lcos/local/packaged_task.hpp>
#include <hpx/util/bind_action.hpp>
#include <hpx/util/deferred_call.hpp>
#include <hpx/util/move.hpp>
#include <hpx/traits/is_callable.hpp>

#include <boost/utility/enable_if.hpp>
#include <boost/preprocessor/enum.hpp>
#include <boost/preprocessor/enum_params.hpp>
#include <boost/preprocessor/iterate.hpp>
#include <boost/type_traits/is_void.hpp>

namespace hpx { namespace detail
{
    // Defer the evaluation of result_of during the SFINAE checks below
#if defined(__clang__)
    template <typename F, typename Result = typename util::deferred_call_result_of<F>::type>
    struct create_future
    {
        typedef lcos::future<Result> type;
    };
#else
    template <typename F, typename ResultOf = util::deferred_call_result_of<F> >
    struct create_future
    {
        typedef lcos::future<typename ResultOf::type> type;
    };
#endif

    template <typename F>
    BOOST_FORCEINLINE
    typename boost::lazy_enable_if<
        boost::is_reference<typename util::deferred_call_result_of<F()>::type>
      , detail::create_future<F()>
    >::type
    call_sync(F && f, boost::mpl::false_)
    {
        return lcos::make_ready_future(boost::ref(f()));
    }
    
    template <typename F>
    BOOST_FORCEINLINE
    typename boost::lazy_disable_if<
        boost::is_reference<typename util::deferred_call_result_of<F()>::type>
      , detail::create_future<F()>
    >::type
    call_sync(F && f, boost::mpl::false_)
    {
        return lcos::make_ready_future(f());
    }

    template <typename F>
    BOOST_FORCEINLINE typename detail::create_future<F()>::type
    call_sync(F && f, boost::mpl::true_)
    {
        f();
        return lcos::make_ready_future();
    }
}}

#if !defined(HPX_USE_PREPROCESSOR_LIMIT_EXPANSION)
#  include <hpx/preprocessed/async.hpp>
#else

#if defined(__WAVE__) && defined(HPX_CREATE_PREPROCESSED_FILES)
#  pragma wave option(preserve: 1, line: 0, output: "preprocessed/async_" HPX_LIMIT_STR ".hpp")
#endif

///////////////////////////////////////////////////////////////////////////////
namespace hpx
{
    ///////////////////////////////////////////////////////////////////////////
    // Launch the given function or function object asynchronously and return a
    // future allowing to synchronize with the returned result.
    template <typename F>
    typename boost::lazy_enable_if_c<
        traits::detail::is_callable_not_action<
            typename util::decay<F>::type()>::value
     && !traits::is_bound_action<typename util::decay<F>::type>::value
      , detail::create_future<F()>
    >::type
    async(BOOST_SCOPED_ENUM(launch) policy, F && f)
    {
        typedef typename util::deferred_call_result_of<
            F()
        >::type result_type;

        if (policy == launch::sync)
        {
            typedef typename boost::is_void<result_type>::type predicate;
            return detail::call_sync(std::forward<F>(f), predicate());
        }

        lcos::local::futures_factory<result_type()> p(std::forward<F>(f));
        if (detail::has_async_policy(policy))
        {
            p.apply(policy);
            if (policy == launch::fork)
            {
                // make sure this thread is executed last
                hpx::this_thread::yield();
            }
        }
        return p.get_future();
    }

    template <typename F>
    typename boost::lazy_enable_if_c<
        traits::detail::is_callable_not_action<
            typename util::decay<F>::type()>::value
     && !traits::is_bound_action<typename util::decay<F>::type>::value
      , detail::create_future<F()>
    >::type
    async(threads::executor& sched, F && f)
    {
        typedef typename util::deferred_call_result_of<
            F()
        >::type result_type;

        lcos::local::futures_factory<result_type()> p(sched,
            std::forward<F>(f));
        p.apply();
        return p.get_future();
    }

    template <typename F>
    typename boost::lazy_enable_if_c<
        traits::detail::is_callable_not_action<
            typename util::decay<F>::type()>::value
     && !traits::is_bound_action<typename util::decay<F>::type>::value
      , detail::create_future<F()>
    >::type
    async(F && f)
    {
        return async(launch::all, std::forward<F>(f));
    }

    ///////////////////////////////////////////////////////////////////////////
    // Launch the given bound action asynchronously and return a future
    // allowing to synchronize with the returned result.
    template <typename Action, typename BoundArgs>
    lcos::future<
        typename hpx::util::detail::bound_action<
            Action, BoundArgs
        >::result_type
    >
    async(hpx::util::detail::bound_action<Action, BoundArgs> const& bound)
    {
        return bound.async();
    }
}

///////////////////////////////////////////////////////////////////////////////
// bring in all N-nary overloads for async
#define BOOST_PP_ITERATION_PARAMS_1                                           \
    (3, (1, HPX_ACTION_ARGUMENT_LIMIT, <hpx/async.hpp>))                      \
    /**/

#include BOOST_PP_ITERATE()

#if defined(__WAVE__) && defined (HPX_CREATE_PREPROCESSED_FILES)
#  pragma wave option(output: null)
#endif

#endif // !defined(HPX_USE_PREPROCESSOR_LIMIT_EXPANSION)

#endif

///////////////////////////////////////////////////////////////////////////////
#else

#define N BOOST_PP_ITERATION()

#define HPX_ASYNC_DECAY_ARG(Z, N, D)                                          \
    typename util::decay<BOOST_PP_CAT(D, N)>::type                            \
    /**/

namespace hpx
{
    ///////////////////////////////////////////////////////////////////////////
    // Define async() overloads for plain local functions and function objects.

    // Launch the given function or function object asynchronously and return a
    // future allowing to synchronize with the returned result.
    template <typename F, BOOST_PP_ENUM_PARAMS(N, typename A)>
    typename boost::lazy_enable_if_c<
        traits::detail::is_callable_not_action<
            typename util::decay<F>::type(
                BOOST_PP_ENUM(N, HPX_ASYNC_DECAY_ARG, A))>::value
     && !traits::is_bound_action<typename util::decay<F>::type>::value
      , detail::create_future<F(BOOST_PP_ENUM_PARAMS(N, A))>
    >::type
    async(BOOST_SCOPED_ENUM(launch) policy, F && f,
        HPX_ENUM_FWD_ARGS(N, A, a))
    {
        typedef typename util::deferred_call_result_of<
            F(BOOST_PP_ENUM_PARAMS(N, A))
        >::type result_type;

        if (policy == launch::sync) {
            typedef typename boost::is_void<result_type>::type predicate;
            return detail::call_sync(
                util::deferred_call(
                    std::forward<F>(f),
                    HPX_ENUM_FORWARD_ARGS(N, A, a)
                ), predicate());
        }
        lcos::local::futures_factory<result_type()> p(
            util::deferred_call(
                std::forward<F>(f),
                HPX_ENUM_FORWARD_ARGS(N, A, a)
            ));
        if (detail::has_async_policy(policy))
        {
            p.apply(policy);
            if (policy == launch::fork)
            {
                // make sure this thread is executed last
                hpx::this_thread::yield();
            }
        }
        return p.get_future();
    }

    template <typename F, BOOST_PP_ENUM_PARAMS(N, typename A)>
    typename boost::lazy_enable_if_c<
        traits::detail::is_callable_not_action<
            typename util::decay<F>::type(
                BOOST_PP_ENUM(N, HPX_ASYNC_DECAY_ARG, A))>::value
     && !traits::is_bound_action<typename util::decay<F>::type>::value
      , detail::create_future<F(BOOST_PP_ENUM_PARAMS(N, A))>
    >::type
    async(threads::executor& sched, F && f,
        HPX_ENUM_FWD_ARGS(N, A, a))
    {
        typedef typename util::deferred_call_result_of<
            F(BOOST_PP_ENUM_PARAMS(N, A))
        >::type result_type;

        lcos::local::futures_factory<result_type()> p(sched,
            util::deferred_call(
                std::forward<F>(f),
                HPX_ENUM_FORWARD_ARGS(N, A, a)
            ));
        p.apply();
        return p.get_future();
    }

    template <typename F, BOOST_PP_ENUM_PARAMS(N, typename A)>
    typename boost::lazy_enable_if_c<
        traits::detail::is_callable_not_action<
            typename util::decay<F>::type(
                BOOST_PP_ENUM(N, HPX_ASYNC_DECAY_ARG, A))>::value
     && !traits::is_bound_action<typename util::decay<F>::type>::value
      , detail::create_future<F(BOOST_PP_ENUM_PARAMS(N, A))>
    >::type
    async(F && f, HPX_ENUM_FWD_ARGS(N, A, a))
    {
        return async(launch::all, std::forward<F>(f),
            HPX_ENUM_FORWARD_ARGS(N, A, a));
    }

    // define async() overloads for bound actions
    template <
        typename Action, typename BoundArgs
      BOOST_PP_COMMA_IF(N) BOOST_PP_ENUM_PARAMS(N, typename A)
    >
    lcos::future<
        typename hpx::util::detail::bound_action<
            Action, BoundArgs
        >::result_type
    >
    async(
        hpx::util::detail::bound_action<Action, BoundArgs> const& bound
      , HPX_ENUM_FWD_ARGS(N, A, a)
    )
    {
        return bound.async(HPX_ENUM_FORWARD_ARGS(N, A, a));
    }
}

#undef HPX_ASYNC_DECAY_ARG
#undef N

#endif

