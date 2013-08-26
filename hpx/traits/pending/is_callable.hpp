//  Copyright (c) 2007-2012 Hartmut Kaiser
//  Copyright (c) 2013 Agustin Berge
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_TRAITS_IS_CALLABLE_APR_15_2012_0601PM)
#define HPX_TRAITS_IS_CALLABLE_APR_15_2012_0601PM

#include <boost/config.hpp>

#if BOOST_VERSION < 105100
#   define BOOST_NO_CXX11_DECLTYPE_N3276
#endif

#if defined(BOOST_NO_SFINAE_EXPR)                                              \
 || defined(BOOST_NO_CXX11_DECLTYPE_N3276)                                     \
 || defined(BOOST_NO_CXX11_VARIADIC_TEMPLATES) // C++03

#include <hpx/config.hpp>

#include <hpx/traits/is_action.hpp>

#include <hpx/util/unused.hpp>
#include <hpx/util/detail/pp_strip_parens.hpp>
#include <hpx/util/detail/remove_reference.hpp>

#include <boost/function_types/components.hpp>
#include <boost/function_types/function_pointer.hpp>

#include <boost/move/move.hpp>

#include <boost/mpl/bool.hpp>
#include <boost/mpl/placeholders.hpp>

#include <boost/preprocessor/cat.hpp>
#include <boost/preprocessor/facilities/intercept.hpp>
#include <boost/preprocessor/repetition/enum_binary_params.hpp>
#include <boost/preprocessor/repetition/enum_params.hpp>
#include <boost/preprocessor/repetition/enum_trailing_params.hpp>
#include <boost/preprocessor/repetition/repeat.hpp>

#include <boost/type_traits/add_const.hpp>
#include <boost/type_traits/add_pointer.hpp>
#include <boost/type_traits/add_reference.hpp>
#include <boost/type_traits/add_rvalue_reference.hpp>
#include <boost/type_traits/add_volatile.hpp>
#include <boost/type_traits/is_class.hpp>
#include <boost/type_traits/is_member_function_pointer.hpp>
#include <boost/type_traits/is_member_object_pointer.hpp>
#include <boost/type_traits/remove_cv.hpp>

#include <boost/utility/enable_if.hpp>
#include <boost/utility/declval.hpp>

#include <cstddef>

// The technique implemented here was devised by Eric Niebler, see:
// http://www.boost.org/doc/libs/1_54_0/doc/html/proto/appendices.html#boost_proto.appendices.implementation.function_arity
namespace hpx { namespace traits
{
    namespace detail
    {
        // creates a type `T` with the (cv-ref)qualifiers of `U`
        template <typename T, typename U>
        struct qualify_as
        {
            typedef T type;
        };

        template <typename T, typename U>
        struct qualify_as<T, U&>
        {
            typedef typename qualify_as<T, U>::type& type;
        };
        template <typename T, typename U>
        struct qualify_as<T, BOOST_FWD_REF(U)>
        {
            typedef BOOST_FWD_REF(HPX_UTIL_STRIP((typename qualify_as<T, U>::type))) type;
        };
    
        template <typename T, typename U>
        struct qualify_as<T, U const>
        {
            typedef typename qualify_as<T, U>::type const type;
        };
        template <typename T, typename U>
        struct qualify_as<T, U volatile>
        {
            typedef typename qualify_as<T, U>::type volatile type;
        };
        template <typename T, typename U>
        struct qualify_as<T, U const volatile>
        {
            typedef typename qualify_as<T, U>::type const volatile type;
        };

        struct fallback_call
        {
            fallback_call const& operator,(int) const volatile;
        };

        template <typename T, std::size_t Arity>
        struct callable_wrapper_fallback;

        template <typename R>
        struct callable_wrapper_fallback<R(), 0>
        {};
        template <typename R>
        struct callable_wrapper_fallback<R(*)(), 0>
        {};
        template <typename R>
        struct callable_wrapper_fallback<R(&)(), 0>
        {};

#       define HPX_TRAITS_DECL_FALLBACK(z, n, data)                             \
        template <typename T>                                                   \
        struct callable_wrapper_fallback<T, n>                                  \
        {                                                                       \
            typedef fallback_call const& (*pointer_to_function)(                \
                    BOOST_PP_ENUM_PARAMS(n, util::unused_type BOOST_PP_INTERCEPT)\
                );                                                              \
            operator pointer_to_function() const volatile;                      \
        };                                                                      \
        /**/

        BOOST_PP_REPEAT(HPX_FUNCTION_ARGUMENT_LIMIT
          , HPX_TRAITS_DECL_FALLBACK, ~);

#       undef HPX_TRAITS_DECL_FALLBACK

        template <typename T, std::size_t Arity, typename Enable = void>
        struct callable_wrapper
          : callable_wrapper_fallback<T, Arity>
        {
            operator typename boost::add_reference<T>::type() const volatile;
        };

        template <typename T, std::size_t Arity>
        struct callable_wrapper<T, Arity
          , typename boost::enable_if<
                boost::is_class<typename util::detail::remove_reference<T>::type>
            >::type
        > : util::detail::remove_reference<T>::type
          , callable_wrapper_fallback<T, Arity>
        {};

        template <typename T, std::size_t Arity>
        struct callable_wrapper<T, Arity
          , typename boost::enable_if<
                boost::is_member_object_pointer<T>
            >::type
        > : callable_wrapper_fallback<T, Arity>
        {
            operator typename boost::function_types::function_pointer<
                    typename boost::function_types::components<
                        T, boost::mpl::_
                    >::type
                >::type() const volatile;

            operator typename boost::function_types::function_pointer<
                    typename boost::function_types::components<
                        T, boost::add_pointer<
                                boost::add_const<boost::add_volatile<boost::mpl::_>>>
                    >::type
                >::type() const volatile;
        };

        template <typename T, std::size_t Arity>
        struct callable_wrapper<T, Arity
          , typename boost::enable_if<
                boost::is_member_function_pointer<T>
            >::type
        > : callable_wrapper_fallback<T, Arity>
        {
            operator typename boost::function_types::function_pointer<
                    typename boost::function_types::components<
                        T, boost::add_reference<boost::mpl::_>
                    >::type
                >::type() const volatile;
            
#           ifndef BOOST_NO_CXX11_RVALUE_REFERENCES
            operator typename boost::function_types::function_pointer<
                    typename boost::function_types::components<
                        T, boost::add_rvalue_reference<boost::mpl::_>
                    >::type
                >::type() const volatile;
#           endif 

            operator typename boost::function_types::function_pointer<
                    typename boost::function_types::components<
                        T, boost::add_pointer<boost::mpl::_>
                    >::type
                >::type() const volatile;
        };

        template <typename T
          , BOOST_PP_ENUM_BINARY_PARAMS(HPX_FUNCTION_ARGUMENT_LIMIT
              , typename A, = void BOOST_PP_INTERCEPT)
          , typename Dummy = void>
        struct is_callable_impl;

#       define HPX_TRAITS_DECL_IS_CALLABLE_IMPL(z, n, data)                     \
        template <typename F BOOST_PP_ENUM_TRAILING_PARAMS(n, typename T)>      \
        struct is_callable_impl<F BOOST_PP_ENUM_TRAILING_PARAMS(n, T)>          \
        {                                                                       \
            typedef typename qualify_as<                                        \
                    detail::callable_wrapper<F, n>, F                           \
                >::type callable_wrapper;                                       \
            typedef char (&no_type)[1];                                         \
            typedef char (&yes_type)[2];                                        \
                                                                                \
            template<typename T>                                                \
            static yes_type can_be_called(T const &);                           \
            static no_type can_be_called(detail::fallback_call const &);        \
                                                                                \
            static bool const value =                                           \
                sizeof(can_be_called((boost::declval<callable_wrapper>()(       \
                    BOOST_PP_ENUM_BINARY_PARAMS(n,                              \
                        boost::declval<T, >() BOOST_PP_INTERCEPT)), 0))         \
                ) == sizeof(yes_type);                                          \
            typedef boost::mpl::bool_<value> type;                              \
        };                                                                      \
        /**/

        BOOST_PP_REPEAT(HPX_FUNCTION_ARGUMENT_LIMIT
          , HPX_TRAITS_DECL_IS_CALLABLE_IMPL, ~);

#       undef HPX_TRAITS_DECL_IS_CALLABLE_IMPL
    }

    template <typename T
      , BOOST_PP_ENUM_BINARY_PARAMS(HPX_FUNCTION_ARGUMENT_LIMIT
          , typename A, = void BOOST_PP_INTERCEPT)>
    struct is_callable
      : detail::is_callable_impl<
            T, BOOST_PP_ENUM_PARAMS(HPX_FUNCTION_ARGUMENT_LIMIT, A)
        >
    {};
}}

#else // C++11

#include <hpx/util/always_void.hpp>

#include <boost/mpl/bool.hpp>

#include <boost/utility/declval.hpp>

namespace hpx { namespace traits
{
    namespace detail
    {
        template <typename T, typename Args, typename Enable = void>
        struct is_callable_impl
          : boost::mpl::false_
        {};

        template <typename T, typename... A>
        struct is_callable_impl<T, void(A...)
          , typename util::always_void<
                decltype(boost::declval<T>()(boost::declval<A>()...))
            >::type
        > : boost::mpl::true_
        {};

        template <typename T, typename C>
        struct is_callable_impl<T, void(C)
            , typename util::always_void<
                  decltype((boost::declval<C>().*boost::declval<T>()))
              >::type
        > : boost::mpl::true_
        {};
        template <typename T, typename C>
        struct is_callable_impl<T, void(C)
            , typename util::always_void<
                  decltype((boost::declval<C>()->*boost::declval<T>()))
              >::type
        > : boost::mpl::true_
        {};

        template <typename T, typename C, typename... A>
        struct is_callable_impl<T, void(C, A...)
            , typename util::always_void<
                  decltype((boost::declval<C>()
                              .*boost::declval<T>())(boost::declval<A>()...))
              >::type
        > : boost::mpl::true_
        {};
        template <typename T, typename C, typename... A>
        struct is_callable_impl<T, void(C, A...)
            , typename util::always_void<
                  decltype((boost::declval<C>()
                              ->*boost::declval<T>())(boost::declval<A>()...))
              >::type
        > : boost::mpl::true_
        {};
    }

    template <typename T, typename... A>
    struct is_callable
      : detail::is_callable_impl<T, void(A...)>
    {};
}}
//
//#else // C++14
//
//#include <hpx/util/always_void.hpp>
//
//#include <boost/mpl/bool.hpp>
//
//#include <boost/type_traits/add_rvalue_reference.hpp>
//
//#include <boost/utility/declval.hpp>
//#include <boost/utility/result_of.hpp>
//
//namespace hpx { namespace traits
//{
//    namespace detail
//    {
//        template <typename T, typename Args, typename Enable = void>
//        struct is_callable_impl
//          : boost::mpl::false_
//        {};
//
//        template <typename T, typename... A>
//        struct is_callable_impl<T, void(A...)
//          , typename util::always_void<
//                typename boost::result_of<
//                    typename boost::add_rvalue_reference<T>::type(A...)
//                >::type
//            >::type
//        > : boost::mpl::true_
//        {};
//
//        // Note: boost::result_of differs form std::result_of,
//        // ignoring member-object-ptrs
//        template <typename T, typename C>
//        struct is_callable_impl<T, void(C)
//            , typename util::always_void<
//                  decltype((boost::declval<C>().*boost::declval<T>()))
//              >::type
//        > : boost::mpl::true_
//        {};
//        template <typename T, typename C>
//        struct is_callable_impl<T, void(C)
//            , typename util::always_void<
//                  decltype((boost::declval<C>()->*boost::declval<T>()))
//              >::type
//        > : boost::mpl::true_
//        {};
//    }
//
//    template <typename T, typename... A>
//    struct is_callable
//      : detail::is_callable_impl<T, void(A...)>
//    {};
//}}

#endif

#include <hpx/config.hpp>

#include <hpx/traits/is_action.hpp>
#include <hpx/util/detail/remove_reference.hpp>

#include <boost/mpl/and.hpp>
#include <boost/mpl/not.hpp>

#include <boost/preprocessor/facilities/intercept.hpp>
#include <boost/preprocessor/repetition/enum_binary_params.hpp>
#include <boost/preprocessor/repetition/enum_trailing_params.hpp>
#include <boost/preprocessor/repetition/repeat.hpp>

namespace hpx { namespace traits { namespace detail
{
    template <typename T
      , BOOST_PP_ENUM_BINARY_PARAMS(HPX_FUNCTION_ARGUMENT_LIMIT
          , typename A, = void BOOST_PP_INTERCEPT)
      , typename Dummy = void>
    struct is_callable_not_action;

#   define HPX_TRAITS_DECL_IS_CALLABLE_NOT_ACTION(z, n, data)                   \
    template <typename T BOOST_PP_ENUM_TRAILING_PARAMS(n, typename A)>          \
    struct is_callable_not_action<T BOOST_PP_ENUM_TRAILING_PARAMS(n, A)>        \
      : boost::mpl::and_<                                                       \
            is_callable<T BOOST_PP_ENUM_TRAILING_PARAMS(n, A)>                  \
          , boost::mpl::not_<traits::is_action<                                 \
                typename util::detail::remove_reference<T>::type>>              \
        >                                                                       \
    {};                                                                         \
    /**/
    
    BOOST_PP_REPEAT(HPX_FUNCTION_ARGUMENT_LIMIT
      , HPX_TRAITS_DECL_IS_CALLABLE_NOT_ACTION, ~);

#   undef HPX_TRAITS_DECL_IS_CALLABLE_NOT_ACTION
}}}

#endif
