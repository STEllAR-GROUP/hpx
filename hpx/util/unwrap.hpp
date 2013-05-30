//  Copyright (c)      2013 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_UTIL_UNWRAP_HPP
#define HPX_UTIL_UNWRAP_HPP

namespace hpx { namespace util {
    namespace detail
    {
        template <typename Future, typename IsFutureRange = typename traits::is_future_range<Future>::type>
        struct unwrap_param_type;

        template <typename Future>
        struct unwrap_param_type<Future, boost::mpl::true_>
        {
            typedef
                std::vector<
                    typename hpx::lcos::future_traits<
                        typename boost::remove_const<
                            typename detail::remove_reference<
                                Future
                            >::type
                        >::type::value_type
                    >::value_type
                >
                type;
        };

        template <typename Future>
        struct unwrap_param_type<Future, boost::mpl::false_>
        {
            typedef
                typename hpx::lcos::future_traits<Future>::value_type
                type;
        };

        template <typename F>
        struct unwrap_impl
        {
            F f_;

            template <typename Sig>
            struct result;

#define     HPX_UTIL_UNWRAP_IMPL_RESULT_OF(Z, N, D)                             \
            typename unwrap_param_type<BOOST_PP_CAT(A, N)>::type                \
            /**/

#define     HPX_UTIL_UNWRAP_IMPL_INVOKE(Z, N, D)                                \
            this->get(BOOST_PP_CAT(a, N))                                       \
            /**/

            template <typename Future>
            typename boost::disable_if<
                typename traits::is_future_range<Future>::type
              , typename unwrap_param_type<Future>::type
            >::type
            get(Future & f)
            {
                return f.get();
            }

            template <typename Range>
            typename boost::enable_if<
                typename traits::is_future_range<Range>::type
              , typename unwrap_param_type<Range>::type
            >::type
            get(Range & r)
            {
                typename unwrap_param_type<Range>::type res;
                
                res.reserve(r.size());

                BOOST_FOREACH(typename Range::value_type const & f, r)
                {
                    res.push_back(f.get());
                }

                return res;
            }

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
            BOOST_PP_REPEAT_FROM_TO(
                1
              , HPX_FUNCTION_ARGUMENT_LIMIT
              , HPX_UTIL_UNWRAP_IMPL_OPERATOR
              , _
            )
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
