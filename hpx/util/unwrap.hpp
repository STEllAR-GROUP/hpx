//  Copyright (c)      2013 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_UTIL_UNWRAP_HPP
#define HPX_UTIL_UNWRAP_HPP

namespace hpx { namespace util {
    namespace detail
    {
        template <typename F>
        struct unwrap_impl
        {
            F f_;

            template <typename Sig>
            struct result;

#define     HPX_UTIL_UNWRAP_IMPL_RESULT_OF(Z, N, D)                             \
            typename hpx::lcos::future_traits<BOOST_PP_CAT(A, N)>::value_type   \
            /**/
#define     HPX_UTIL_UNWRAP_IMPL_INVOKE(Z, N, D)                                \
            BOOST_PP_CAT(a, N).get()                                            \
            /**/

#define     HPX_UTIL_UNWRAP_IMPL_OPERATOR(Z, N, D)                              \
            template <typename This, BOOST_PP_ENUM_PARAMS(N, typename A)>       \
            struct result<This const(BOOST_PP_ENUM_PARAMS(N, A))>               \
            {                                                                   \
                typedef typename                                                \
                    boost::result_of<                                           \
                        F const(BOOST_PP_ENUM(N, HPX_UTIL_UNWRAP_IMPL_RESULT_OF, _))\
                    >::type                                                     \
                    type;                                                       \
            };                                                                  \
                                                                                \
            template <BOOST_PP_ENUM_PARAMS(N, typename A)>                      \
            typename result<unwrap_impl const(BOOST_PP_ENUM_PARAMS(N, A))>::type\
            operator()(HPX_ENUM_FWD_ARGS(N, A, a)) const                        \
            {                                                                   \
                return f_(BOOST_PP_ENUM(N, HPX_UTIL_UNWRAP_IMPL_INVOKE, _));    \
            }                                                                   \
                                                                                \
            template <typename This, BOOST_PP_ENUM_PARAMS(N, typename A)>       \
            struct result<This(BOOST_PP_ENUM_PARAMS(N, A))>                     \
            {                                                                   \
                typedef typename                                                \
                    boost::result_of<                                           \
                        F(BOOST_PP_ENUM(N, HPX_UTIL_UNWRAP_IMPL_RESULT_OF, _))  \
                    >::type                                                     \
                    type;                                                       \
            };                                                                  \
                                                                                \
            template <BOOST_PP_ENUM_PARAMS(N, typename A)>                      \
            typename result<unwrap_impl(BOOST_PP_ENUM_PARAMS(N, A))>::type      \
            operator()(HPX_ENUM_FWD_ARGS(N, A, a))                              \
            {                                                                   \
                return f_(BOOST_PP_ENUM(N, HPX_UTIL_UNWRAP_IMPL_INVOKE, _));    \
            }                                                                   \
            /**/
            BOOST_PP_REPEAT_FROM_TO(1, 15, HPX_UTIL_UNWRAP_IMPL_OPERATOR, _)
#undef      HPX_UTIL_UNWRAP_IMPL_OPERATOR
        };
    }

    template <typename F>
    detail::unwrap_impl<typename detail::remove_reference<F>::type >
    unwrap(BOOST_FWD_REF(F) f)
    {
        detail::unwrap_impl<typename detail::remove_reference<F>::type >
            res = {boost::forward<F>(f)};

        return res;
    }
}}

#endif
