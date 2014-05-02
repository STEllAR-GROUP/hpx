//  Copyright (c)      2013 Thomas Heller
//  Copyright (c)      2014 Agustin Berge
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !BOOST_PP_IS_ITERATING

#ifndef HPX_UTIL_UNWRAPPED_HPP
#define HPX_UTIL_UNWRAPPED_HPP

#include <hpx/lcos/future.hpp>
#include <hpx/traits/is_future.hpp>
#include <hpx/traits/is_future_range.hpp>
#include <hpx/traits/is_future_tuple.hpp>
#include <hpx/util/decay.hpp>
#include <hpx/util/invoke_fused.hpp>
#include <hpx/util/result_of.hpp>
#include <hpx/util/tuple.hpp>

#include <boost/mpl/eval_if.hpp>
#include <boost/fusion/include/fold.hpp>
#include <boost/preprocessor/cat.hpp>
#include <boost/preprocessor/iteration/iterate.hpp>
#include <boost/preprocessor/repetition/enum.hpp>
#include <boost/preprocessor/repetition/enum_params.hpp>
#include <boost/preprocessor/repetition/repeat_from_to.hpp>
#include <boost/utility/enable_if.hpp>

namespace hpx { namespace util
{
    namespace detail
    {
        template <typename T, typename Enable = void>
        struct unwrap_impl;

        ///////////////////////////////////////////////////////////////////////
        template <typename T>
        struct unwrap_impl<
            T,
            typename boost::enable_if<traits::is_future<T> >::type
        >
        {
            typedef typename traits::future_traits<T>::type value_type;
            typedef typename boost::is_void<value_type>::type is_void;

            typedef typename boost::mpl::if_<
                is_void, void, value_type
            >::type type;

            template <typename Future>
            static type call(Future& future, /*is_void=*/boost::mpl::false_)
            {
                return future.get();
            }

            template <typename Future>
            static type call(Future& future, /*is_void=*/boost::mpl::true_)
            {
                future.get();
            }

            template <typename Future>
            static type call(Future&& future)
            {
                return call(future, is_void());
            }
        };

        ///////////////////////////////////////////////////////////////////////
        template <typename T>
        struct unwrap_impl<
            T,
            typename boost::enable_if<traits::is_future_range<T> >::type
        >
        {
            typedef typename T::value_type future_type;
            typedef typename traits::future_traits<future_type>::type value_type;
            typedef typename boost::is_void<value_type>::type is_void;

            typedef typename boost::mpl::if_<
                is_void, void, std::vector<value_type>
            >::type type;

            template <typename Range>
            static type call(Range& range, /*is_void=*/boost::mpl::false_)
            {
                type result;
                BOOST_FOREACH(typename Range::value_type & f, range)
                {
                    result.push_back(unwrap_impl<future_type>::call(f));
                }

                return result;
            }

            template <typename Range>
            static type call(Range& range, /*is_void=*/boost::mpl::true_)
            {
                BOOST_FOREACH(typename Range::value_type & f, range)
                {
                    unwrap_impl<future_type>::call(f);
                }
            }

            template <typename Range>
            static type call(Range&& range)
            {
                return call(range, is_void());
            }
        };

        ///////////////////////////////////////////////////////////////////////
        template <typename Tuple, typename U>
        struct unwrap_tuple_push_back;

        template <typename U>
        struct unwrap_tuple_push_back<util::tuple<>, U>
        {
            typedef util::tuple<U> type;

            template <typename U_>
            static type call(util::tuple<>, U_&& value)
            {
                return type(std::forward<U_>(value));
            }
        };

#       define UNWRAP_TUPLE_PUSH_BACK_GET(Z, N, D)                            \
        util::get<N>(std::forward<Tuple_>(tuple))                             \
        /**/
#       define UNWRAP_TUPLE_PUSH_BACK(Z, N, D)                                \
        template <BOOST_PP_ENUM_PARAMS(N, typename T), typename U>            \
        struct unwrap_tuple_push_back<                                        \
            util::tuple<BOOST_PP_ENUM_PARAMS(N, T)>, U                        \
        >                                                                     \
        {                                                                     \
            typedef util::tuple<BOOST_PP_ENUM_PARAMS(N, T), U> type;          \
                                                                              \
            template <typename Tuple_, typename U_>                           \
            static type call(Tuple_&& tuple, U_&& value)                      \
            {                                                                 \
                return type(                                                  \
                    BOOST_PP_ENUM(N, UNWRAP_TUPLE_PUSH_BACK_GET, _)           \
                  , std::forward<U_>(value));                                 \
            }                                                                 \
        };                                                                    \
        /**/

        BOOST_PP_REPEAT_FROM_TO(
            1, HPX_TUPLE_LIMIT
          , UNWRAP_TUPLE_PUSH_BACK, _
        );

#       undef UNWRAP_TUPLE_PUSH_BACK_GET
#       undef UNWRAP_TUPLE_PUSH_BACK

        struct unwrap_tuple_impl
        {
            template <typename>
            struct result;

            template <typename This, typename Tuple, typename Future>
            struct result<This(Tuple, Future)>
              : boost::mpl::eval_if<
                    typename unwrap_impl<
                        typename util::decay<Future>::type
                    >::is_void
                  , util::decay<Tuple>
                  , unwrap_tuple_push_back<
                        typename util::decay<Tuple>::type
                      , typename unwrap_impl<
                            typename util::decay<Future>::type
                        >::type
                    >
                >
            {};

            template <typename Tuple, typename Future>
            typename result<unwrap_tuple_impl(Tuple, Future)>::type
            operator()(Tuple tuple, Future&& f, typename boost::disable_if<
                typename unwrap_impl<
                    typename util::decay<Future>::type>::is_void>::type* = 0
            ) const
            {
                typedef
                    unwrap_impl<typename util::decay<Future>::type>
                    unwrap_impl_t;

                typedef
                    unwrap_tuple_push_back<
                        typename util::decay<Tuple>::type
                      , typename unwrap_impl_t::type
                    >
                    unwrap_tuple_push_back_t;

                return unwrap_tuple_push_back_t::call(
                    std::move(tuple), unwrap_impl_t::call(f));
            }

            template <typename Tuple, typename Future>
            typename result<unwrap_tuple_impl(Tuple, Future)>::type
            operator()(Tuple tuple, Future&& f, typename boost::enable_if<
                typename unwrap_impl<
                    typename util::decay<Future>::type>::is_void>::type* = 0
            ) const
            {
                typedef
                    unwrap_impl<typename util::decay<Future>::type>
                    unwrap_impl_t;

                unwrap_impl_t::call(f);
                return std::move(tuple);
            }
        };

        template <typename T>
        struct unwrap_impl<
            T,
            typename boost::enable_if<traits::is_future_tuple<T> >::type
        >
        {
            typedef typename boost::fusion::result_of::fold<
                T, util::tuple<>, unwrap_tuple_impl
            >::type type;

            template <typename Tuple>
            static type call(Tuple&& tuple)
            {
                return boost::fusion::fold(
                    tuple, util::tuple<>(), unwrap_tuple_impl());
            }
        };

        ///////////////////////////////////////////////////////////////////////
        template <typename F, typename T,
            typename TD = typename decay<T>::type, typename Enable = void>
        struct unwrapped_impl_result
        {};

        template <typename F, typename T, typename TD>
        struct unwrapped_impl_result<
            F, T, TD,
            typename boost::enable_if<traits::is_future<TD> >::type
        > : util::invoke_fused_result_of<
                F(typename unwrap_impl<util::tuple<TD> >::type)
            >
        {};

        template <typename F, typename T, typename TD>
        struct unwrapped_impl_result<
            F, T, TD,
            typename boost::enable_if<traits::is_future_range<TD> >::type
        > : util::invoke_fused_result_of<
                F(util::tuple<typename unwrap_impl<TD>::type>)
            >
        {};

        template <typename F, typename T, typename TD>
        struct unwrapped_impl_result<
            F, T, TD,
            typename boost::enable_if<traits::is_future_tuple<TD> >::type
        > : util::invoke_fused_result_of<
                F(typename unwrap_impl<TD>::type)
            >
        {};

        template <typename F>
        struct unwrapped_impl
        {
            explicit unwrapped_impl(F const& f)
              : f_(f)
            {}

            explicit unwrapped_impl(F&& f)
              : f_(std::move(f))
            {}

            template <typename Sig>
            struct result;

            template <typename This>
            struct result<This()>
              : util::detail::result_of_or<F(), hpx::util::unused_type>
            {};

            BOOST_FORCEINLINE
            typename result<unwrapped_impl()>::type
            operator()()
            {
                typedef typename result<unwrapped_impl()>::type result_type;

                return util::invoke_fused_r<result_type>(f_,
                    util::make_tuple());
            }

            template <typename This, typename T0>
            struct result<This(T0)>
              : unwrapped_impl_result<F, T0>
            {};

            // future
            template <typename T0>
            BOOST_FORCEINLINE
            typename boost::lazy_enable_if_c<
                traits::is_future<typename decay<T0>::type>::value
              , result<unwrapped_impl(T0)>
            >::type operator()(T0&& t0)
            {
                typedef typename result<unwrapped_impl(T0)>::type result_type;
                typedef
                    unwrap_impl<util::tuple<typename decay<T0>::type> >
                    unwrap_impl_t;

                return util::invoke_fused_r<result_type>(f_,
                    unwrap_impl_t::call(util::forward_as_tuple(t0)));
            }

            // future-range
            template <typename T0>
            BOOST_FORCEINLINE
            typename boost::lazy_enable_if_c<
                traits::is_future_range<typename decay<T0>::type>::value
             && !unwrap_impl<typename decay<T0>::type>::is_void::value
              , result<unwrapped_impl(T0)>
            >::type operator()(T0&& t0)
            {
                typedef typename result<unwrapped_impl(T0)>::type result_type;
                typedef
                    unwrap_impl<typename decay<T0>::type>
                    unwrap_impl_t;

                return util::invoke_fused_r<result_type>(f_,
                    util::forward_as_tuple(unwrap_impl_t::call(t0)));
            }

            template <typename T0>
            BOOST_FORCEINLINE
            typename boost::lazy_enable_if_c<
                traits::is_future_range<typename decay<T0>::type>::value
             && unwrap_impl<typename decay<T0>::type>::is_void::value
              , result<unwrapped_impl(T0)>
            >::type operator()(T0&& t0)
            {
                typedef typename result<unwrapped_impl(T0)>::type result_type;
                typedef
                    unwrap_impl<typename decay<T0>::type>
                    unwrap_impl_t;

                unwrap_impl_t::call(t0);
                return util::invoke_fused_r<result_type>(f_,
                    util::forward_as_tuple());
            }

            // future-tuple
            template <typename T0>
            BOOST_FORCEINLINE
            typename boost::lazy_enable_if_c<
                traits::is_future_tuple<typename decay<T0>::type>::value
              , result<unwrapped_impl(T0)>
            >::type operator()(T0&& t0)
            {
                typedef typename result<unwrapped_impl(T0)>::type result_type;
                typedef
                    unwrap_impl<typename decay<T0>::type>
                    unwrap_impl_t;

                return util::invoke_fused_r<result_type>(f_,
                    unwrap_impl_t::call(t0));
            }

            // futures
#define     HPX_UTIL_UNWRAPPED_DECAY(Z, N, D)                                 \
            typename util::decay<BOOST_PP_CAT(T, N)>::type                    \
            /**/
#           define HPX_UTIL_UNWRAP_IMPL_OPERATOR(Z, N, D)                     \
            template <typename This, BOOST_PP_ENUM_PARAMS(N, typename T)>     \
            struct result<This(BOOST_PP_ENUM_PARAMS(N, T))>                   \
              : result<This(util::tuple<                                      \
                    BOOST_PP_ENUM(N, HPX_UTIL_UNWRAPPED_DECAY, _)>)>          \
            {};                                                               \
                                                                              \
            template <BOOST_PP_ENUM_PARAMS(N, typename T)>                    \
            BOOST_FORCEINLINE                                                 \
            typename result<unwrapped_impl(BOOST_PP_ENUM_PARAMS(N, T))>::type \
            operator()(HPX_ENUM_FWD_ARGS(N, T, t))                            \
            {                                                                 \
                typedef                                                       \
                    typename result<unwrapped_impl(BOOST_PP_ENUM_PARAMS(N, T))>::type\
                    result_type;                                              \
                typedef                                                       \
                    unwrap_impl<util::tuple<                                  \
                        BOOST_PP_ENUM(N, HPX_UTIL_UNWRAPPED_DECAY, _)> >      \
                    unwrap_impl_t;                                            \
                                                                              \
                return util::invoke_fused_r<result_type>(f_,                  \
                    unwrap_impl_t::call(util::forward_as_tuple(               \
                        HPX_ENUM_FORWARD_ARGS(N, T, t))));                    \
            }                                                                 \
            /**/

            BOOST_PP_REPEAT_FROM_TO(
                2, HPX_FUNCTION_ARGUMENT_LIMIT
              , HPX_UTIL_UNWRAP_IMPL_OPERATOR, _
            );

#           undef HPX_UTIL_UNWRAPPED_DECAY
#           undef HPX_UTIL_UNWRAP_IMPL_OPERATOR

            F f_;
        };
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename Future>
    typename boost::lazy_enable_if_c<
        traits::is_future<typename decay<Future>::type>::value
     || traits::is_future_range<typename decay<Future>::type>::value
     || traits::is_future_tuple<typename decay<Future>::type>::value
      , detail::unwrap_impl<typename decay<Future>::type>
    >::type unwrapped(Future&& f)
    {
        typedef
            detail::unwrap_impl<typename decay<Future>::type>
            unwrap_impl_t;

        return unwrap_impl_t::call(std::forward<Future>(f));
    }

    template <typename F>
    typename boost::disable_if_c<
        traits::is_future<typename decay<F>::type>::value
     || traits::is_future_range<typename decay<F>::type>::value
     || traits::is_future_tuple<typename decay<F>::type>::value
      , detail::unwrapped_impl<typename util::decay<F>::type >
    >::type unwrapped(F && f)
    {
        detail::unwrapped_impl<typename util::decay<F>::type >
            res(std::forward<F>(f));

        return std::move(res);
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename Future>
    typename boost::lazy_enable_if_c<
        traits::is_future<typename decay<Future>::type>::value
     || traits::is_future_range<typename decay<Future>::type>::value
     || traits::is_future_tuple<typename decay<Future>::type>::value
      , detail::unwrap_impl<typename detail::unwrap_impl<
            typename decay<Future>::type
        > >
    >::type unwrapped2(Future&& f)
    {
        return unwrapped(unwrapped(std::forward<Future>(f)));
    }

    template <typename F>
    typename boost::disable_if_c<
        traits::is_future<typename decay<F>::type>::value
     || traits::is_future_range<typename decay<F>::type>::value
     || traits::is_future_tuple<typename decay<F>::type>::value
      , detail::unwrapped_impl<detail::unwrapped_impl<
            typename util::decay<F>::type
        > >
    >::type unwrapped2(F && f)
    {
        typedef detail::unwrapped_impl<detail::unwrapped_impl<
            typename util::decay<F>::type
        > > result_type;

        detail::unwrapped_impl<typename util::decay<F>::type >
            res(std::forward<F>(f));

        return result_type(std::move(res));
    }
}}

#if !defined(HPX_USE_PREPROCESSOR_LIMIT_EXPANSION)
#  include <hpx/util/preprocessed/unwrapped.hpp>
#else

#if defined(__WAVE__) && defined(HPX_CREATE_PREPROCESSED_FILES)
#  pragma wave option(preserve: 1, line: 0, output: "preprocessed/unwrapped_" HPX_LIMIT_STR ".hpp")
#endif

#define BOOST_PP_ITERATION_PARAMS_1                                           \
    (3, (2, HPX_FUNCTION_ARGUMENT_LIMIT, <hpx/util/unwrapped.hpp>))           \
/**/
#include BOOST_PP_ITERATE()

#if defined(__WAVE__) && defined (HPX_CREATE_PREPROCESSED_FILES)
#  pragma wave option(output: null)
#endif

#endif // !defined(HPX_USE_PREPROCESSOR_LIMIT_EXPANSION)

#endif

///////////////////////////////////////////////////////////////////////////////
#else // BOOST_PP_IS_ITERATING

#define N BOOST_PP_ITERATION()

#define HPX_UTIL_UNWRAPPED_DECAY(Z, N, D)                                     \
    typename util::decay<BOOST_PP_CAT(T, N)>::type                            \
    /**/

namespace hpx { namespace util
{
    ///////////////////////////////////////////////////////////////////////////
    template <BOOST_PP_ENUM_PARAMS(N, typename T)>
    typename boost::lazy_enable_if_c<
        traits::is_future_tuple<util::tuple<
            BOOST_PP_ENUM(N, HPX_UTIL_UNWRAPPED_DECAY, _)
        > >::value
      , detail::unwrap_impl<util::tuple<
            BOOST_PP_ENUM(N, HPX_UTIL_UNWRAPPED_DECAY, _)
        > >
    >::type unwrapped(HPX_ENUM_FWD_ARGS(N, T, f))
    {
        typedef detail::unwrap_impl<util::tuple<
            BOOST_PP_ENUM(N, HPX_UTIL_UNWRAPPED_DECAY, _)
        > > unwrap_impl_t;

        return unwrap_impl_t::call(util::forward_as_tuple(
            HPX_ENUM_FORWARD_ARGS(N, T, f)));
    }
}}

#undef HPX_UTIL_UNWRAPPED_DECAY
#undef N

#endif
