//  Copyright (c) 2007-2012 Hartmut Kaiser
//  Copyright (c) 2013 Agustin Berge
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_TRAITS_IS_CALLABLE_APR_15_2012_0601PM)
#define HPX_TRAITS_IS_CALLABLE_APR_15_2012_0601PM

#include <hpx/config.hpp>

#if defined(BOOST_NO_SFINAE_EXPR)                                              \
 || defined(BOOST_NO_CXX11_DECLTYPE_N3276)                                     \
 || defined(BOOST_NO_CXX11_VARIADIC_TEMPLATES) // C++03

#include <hpx/traits/is_action.hpp>
#include <hpx/util/decay.hpp>
#include <hpx/util/detail/pp_strip_parens.hpp>
#include <hpx/util/detail/qualify_as.hpp>

#include <boost/function_types/components.hpp>
#include <boost/function_types/function_pointer.hpp>
#include <boost/mpl/bool.hpp>
#include <boost/preprocessor/cat.hpp>
#include <boost/preprocessor/facilities/intercept.hpp>
#include <boost/preprocessor/repetition/enum_binary_params.hpp>
#include <boost/preprocessor/repetition/enum_params.hpp>
#include <boost/preprocessor/repetition/enum_trailing_params.hpp>
#include <boost/preprocessor/repetition/repeat.hpp>
#include <boost/ref.hpp>
#include <boost/type_traits/has_dereference.hpp>
#include <boost/type_traits/is_class.hpp>
#include <boost/utility/enable_if.hpp>
#include <boost/utility/declval.hpp>

#include <cstddef>
#include <utility>

// The technique implemented here was devised by Eric Niebler, see:
// http://www.boost.org/doc/libs/1_54_0/doc/html/proto/appendices.html#boost_proto.appendices.implementation.function_arity
namespace hpx { namespace traits
{
    namespace detail
    {
        ///////////////////////////////////////////////////////////////////////
        struct fallback_argument
        {
            template <typename T> fallback_argument(T const&);
        };

        struct fallback_call
        {
            fallback_call const& operator,(int) const volatile;
        };

        ///////////////////////////////////////////////////////////////////////
        template <typename T, std::size_t Arity>
        struct callable_wrapper_fallback;

        template <typename R>
        struct callable_wrapper_fallback<R(*)(), 0>
        {};

#       define HPX_TRAITS_DECL_FALLBACK(Z, N, D)                              \
        template <typename T>                                                 \
        struct callable_wrapper_fallback<T, N>                                \
        {                                                                     \
            typedef fallback_call const& (*pointer_to_function)(              \
                    BOOST_PP_ENUM_PARAMS(N, fallback_argument BOOST_PP_INTERCEPT)\
                );                                                            \
            operator pointer_to_function() const volatile;                    \
        };                                                                    \
        /**/

        BOOST_PP_REPEAT(
            HPX_FUNCTION_ARGUMENT_LIMIT
          , HPX_TRAITS_DECL_FALLBACK, _
        );

#       undef HPX_TRAITS_DECL_FALLBACK

        ///////////////////////////////////////////////////////////////////////
        template <typename T, std::size_t Arity, typename Enable = void>
        struct callable_wrapper
          : callable_wrapper_fallback<T, Arity>
        {};

        template <typename T, std::size_t Arity>
        struct callable_wrapper<T, Arity
          , typename boost::enable_if_c<
                boost::is_class<T>::value
             && !boost::is_reference_wrapper<T>::value
            >::type
        > : T
          , callable_wrapper_fallback<T, Arity>
        {};

        template <typename T, std::size_t Arity>
        struct callable_wrapper<boost::reference_wrapper<T>, Arity>
          : callable_wrapper<typename util::decay<T>::type, Arity>
        {};

#       define HPX_TRAITS_DECL_CALLABLE_WRAPPER_FUNCTION(Z, N, D)             \
        template <typename R                                                  \
            BOOST_PP_ENUM_TRAILING_PARAMS(N, typename P)                      \
          , std::size_t Arity>                                                \
        struct callable_wrapper<R(*)(BOOST_PP_ENUM_PARAMS(N, P)), Arity>      \
          : callable_wrapper_fallback<R(*)(BOOST_PP_ENUM_PARAMS(N, P)), Arity>\
        {                                                                     \
            R operator()(BOOST_PP_ENUM_PARAMS(N, P)) const volatile;          \
        };                                                                    \
                                                                              \
        template <typename R, typename C                                      \
            BOOST_PP_ENUM_TRAILING_PARAMS(N, typename P)                      \
          , std::size_t Arity>                                                \
        struct callable_wrapper<R(C::*)(BOOST_PP_ENUM_PARAMS(N, P)), Arity>   \
          : callable_wrapper_fallback<R(C::*)(BOOST_PP_ENUM_PARAMS(N, P)), Arity>\
        {                                                                     \
            R operator()(C*                                                   \
                BOOST_PP_ENUM_TRAILING_PARAMS(N, P)) const volatile;          \
            R operator()(C&                                                   \
                BOOST_PP_ENUM_TRAILING_PARAMS(N, P)) const volatile;          \
            R operator()(C&&                                                  \
                BOOST_PP_ENUM_TRAILING_PARAMS(N, P)) const volatile;          \
            template <typename T>                                             \
            typename boost::enable_if_c<                                      \
                boost::has_dereference<T, C&>::value                          \
              , R                                                             \
            >::type operator()(T                                              \
                BOOST_PP_ENUM_TRAILING_PARAMS(N, P)) const volatile;          \
        };                                                                    \
                                                                              \
        template <typename R, typename C                                      \
            BOOST_PP_ENUM_TRAILING_PARAMS(N, typename P)                      \
          , std::size_t Arity>                                                \
        struct callable_wrapper<R(C::*)(BOOST_PP_ENUM_PARAMS(N, P)) const, Arity>\
          : callable_wrapper_fallback<R(C::*)(BOOST_PP_ENUM_PARAMS(N, P)) const, Arity>\
        {                                                                     \
            R operator()(C const*                                             \
                BOOST_PP_ENUM_TRAILING_PARAMS(N, P)) const volatile;          \
            R operator()(C const&                                             \
                BOOST_PP_ENUM_TRAILING_PARAMS(N, P)) const volatile;          \
            R operator()(C const&&                                            \
                BOOST_PP_ENUM_TRAILING_PARAMS(N, P)) const volatile;          \
            template <typename T>                                             \
            typename boost::enable_if_c<                                      \
                boost::has_dereference<T, C const&>::value                    \
              , R                                                             \
            >::type operator()(T                                              \
                BOOST_PP_ENUM_TRAILING_PARAMS(N, P)) const volatile;          \
        };                                                                    \
        /**/

        BOOST_PP_REPEAT(
            HPX_FUNCTION_ARGUMENT_LIMIT
          , HPX_TRAITS_DECL_CALLABLE_WRAPPER_FUNCTION, _
        );

#       undef HPX_TRAITS_DECL_CALLABLE_WRAPPER_FUNCTION

        template <typename M, typename C, std::size_t Arity>
        struct callable_wrapper<M C::*, Arity>
          : callable_wrapper_fallback<M C::*, Arity>
        {
            M& operator()(C const*) const volatile;
            M& operator()(C) const volatile;
            template <typename T>
            typename boost::enable_if_c<
                boost::has_dereference<T, C>::value
              , M&
            >::type operator()(T) const volatile;
        };

        ///////////////////////////////////////////////////////////////////////
        template <typename T
          , BOOST_PP_ENUM_BINARY_PARAMS(HPX_FUNCTION_ARGUMENT_LIMIT
              , typename A, = void BOOST_PP_INTERCEPT)
          , typename Dummy = void>
        struct is_callable_impl;

#       define HPX_TRAITS_DECL_IS_CALLABLE_IMPL(Z, N, D)                      \
        template <typename F BOOST_PP_ENUM_TRAILING_PARAMS(N, typename T)>    \
        struct is_callable_impl<F BOOST_PP_ENUM_TRAILING_PARAMS(N, T)>        \
        {                                                                     \
            typedef typename util::detail::qualify_as<                        \
                    detail::callable_wrapper<typename util::decay<F>::type, N>, F\
                >::type callable_wrapper;                                     \
            typedef char (&no_type)[1];                                       \
            typedef char (&yes_type)[2];                                      \
                                                                              \
            template<typename T>                                              \
            static yes_type can_be_called(T const &);                         \
            static no_type can_be_called(detail::fallback_call const &);      \
                                                                              \
            static bool const value =                                         \
                sizeof(can_be_called((boost::declval<callable_wrapper>()(     \
                    BOOST_PP_ENUM_BINARY_PARAMS(N,                            \
                        boost::declval<T, >() BOOST_PP_INTERCEPT)), 0))       \
                ) == sizeof(yes_type);                                        \
            typedef boost::mpl::bool_<value> type;                            \
        };                                                                    \
        /**/

        BOOST_PP_REPEAT(
            HPX_FUNCTION_ARGUMENT_LIMIT
          , HPX_TRAITS_DECL_IS_CALLABLE_IMPL, _
        );

#       undef HPX_TRAITS_DECL_IS_CALLABLE_IMPL
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename T>
    struct is_callable;

#   define HPX_TRAITS_DECL_IS_CALLABLE(Z, N, D)                               \
    template <typename F BOOST_PP_ENUM_TRAILING_PARAMS(N, typename A)>        \
    struct is_callable<F(BOOST_PP_ENUM_PARAMS(N, A))>                         \
      : detail::is_callable_impl<F BOOST_PP_ENUM_TRAILING_PARAMS(N, A)>       \
    {};                                                                       \
    /**/

    BOOST_PP_REPEAT(
        HPX_FUNCTION_ARGUMENT_LIMIT
      , HPX_TRAITS_DECL_IS_CALLABLE, _
    );

#   undef HPX_TRAITS_DECL_IS_CALLABLE
}}

#else // C++11

#include <hpx/util/always_void.hpp>
#include <hpx/util/decay.hpp>

#include <boost/mpl/bool.hpp>
#include <boost/ref.hpp>
#include <boost/utility/declval.hpp>
#include <boost/utility/enable_if.hpp>

namespace hpx { namespace traits
{
    namespace detail
    {
        template <typename T, typename Enable = void, typename Enable2 = void>
        struct is_callable_impl
          : boost::mpl::false_
        {};

        template <typename F, typename... A>
        struct is_callable_impl<F(A...)
          , typename util::always_void<decltype(
                boost::declval<F>()(boost::declval<A>()...)
            )>::type
        > : boost::mpl::true_
        {};

        template <typename F, typename C>
        struct is_callable_impl<F(C)
          , typename util::always_void<decltype(
                boost::declval<C>().*boost::declval<F>()
            )>::type
        > : boost::mpl::true_
        {};
        template <typename F, typename C>
        struct is_callable_impl<F(C)
          , typename util::always_void<decltype(
                (*boost::declval<C>()).*boost::declval<F>()
            )>::type
        > : boost::mpl::true_
        {};
        template <typename F, typename C>
        struct is_callable_impl<F(C)
          , typename boost::enable_if_c<
                boost::is_reference_wrapper<
                    typename util::decay<C>::type
                >::value
            >::type
          , typename util::always_void<decltype(
                (boost::declval<C>().get()).*boost::declval<F>()
            )>::type
        > : boost::mpl::true_
        {};

        template <typename F, typename C, typename... A>
        struct is_callable_impl<F(C, A...)
          , typename util::always_void<decltype(
                (boost::declval<C>().*boost::declval<F>())
                    (boost::declval<A>()...)
            )>::type
        > : boost::mpl::true_
        {};
        template <typename F, typename C, typename... A>
        struct is_callable_impl<F(C, A...)
          , typename util::always_void<decltype(
                ((*boost::declval<C>()).*boost::declval<F>())
                    (boost::declval<A>()...)
            )>::type
        > : boost::mpl::true_
        {};
        template <typename F, typename C, typename... A>
        struct is_callable_impl<F(C, A...)
          , typename boost::enable_if_c<
                boost::is_reference_wrapper<
                    typename util::decay<C>::type
                >::value
            >::type
          , typename util::always_void<decltype(
                ((boost::declval<C>().get()).*boost::declval<F>())
                    (boost::declval<A>()...)
            )>::type
        > : boost::mpl::true_
        {};

        // support boost::[c]ref, which is not callable as std::[c]ref
        template <typename F, typename... A>
        struct is_callable_impl<F(A...)
          , typename boost::enable_if_c<
                boost::is_reference_wrapper<
                    typename util::decay<F>::type
                >::value
            >::type
        > : is_callable_impl<
                typename boost::unwrap_reference<
                    typename util::decay<F>::type
                >::type&(A...)
            >
        {};
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename T>
    struct is_callable;

    template <typename F, typename... A>
    struct is_callable<F(A...)>
      : detail::is_callable_impl<F(A...)>
    {};
}}

#endif

#include <hpx/traits/is_action.hpp>
#include <hpx/util/decay.hpp>

#include <boost/mpl/and.hpp>
#include <boost/mpl/not.hpp>
#include <boost/preprocessor/facilities/intercept.hpp>
#include <boost/preprocessor/repetition/enum_params.hpp>
#include <boost/preprocessor/repetition/enum_trailing_params.hpp>
#include <boost/preprocessor/repetition/repeat.hpp>

namespace hpx { namespace traits { namespace detail
{
    ///////////////////////////////////////////////////////////////////////////
    template <typename T>
    struct is_callable_not_action;

#   define HPX_TRAITS_DECL_IS_CALLABLE_NOT_ACTION(Z, N, D)                    \
    template <typename T BOOST_PP_ENUM_TRAILING_PARAMS(N, typename A)>        \
    struct is_callable_not_action<T(BOOST_PP_ENUM_PARAMS(N, A))>              \
      : boost::mpl::and_<                                                     \
            is_callable<T(BOOST_PP_ENUM_PARAMS(N, A))>                        \
          , boost::mpl::not_<traits::is_action<typename util::decay<T>::type> >\
        >                                                                     \
    {};                                                                       \
    /**/

    BOOST_PP_REPEAT(
        HPX_FUNCTION_ARGUMENT_LIMIT
      , HPX_TRAITS_DECL_IS_CALLABLE_NOT_ACTION, _
    );

#   undef HPX_TRAITS_DECL_IS_CALLABLE_NOT_ACTION
}}}

#endif
