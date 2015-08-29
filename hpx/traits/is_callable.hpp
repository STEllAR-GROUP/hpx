//  Copyright (c) 2007-2012 Hartmut Kaiser
//  Copyright (c) 2013 Agustin Berge
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_TRAITS_IS_CALLABLE_APR_15_2012_0601PM)
#define HPX_TRAITS_IS_CALLABLE_APR_15_2012_0601PM

#include <hpx/config.hpp>

#if defined(BOOST_NO_SFINAE_EXPR) || defined(BOOST_NO_CXX11_DECLTYPE_N3276) // C++03

#include <hpx/traits/is_action.hpp>
#include <hpx/util/decay.hpp>
#include <hpx/util/detail/pack.hpp>
#include <hpx/util/detail/qualify_as.hpp>

#include <boost/mpl/bool.hpp>
#include <boost/ref.hpp>
#include <boost/type_traits/has_dereference.hpp>
#include <boost/type_traits/is_class.hpp>
#include <boost/utility/enable_if.hpp>
#include <boost/utility/declval.hpp>

#include <cstddef>
#include <utility>

// The technique implemented here was devised by Eric Niebler, see:
// http://www.boost.org/doc/libs/1_54_0/doc/html/proto/appendices.html
// #boost_proto.appendices.implementation.function_arity
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

        template <std::size_t Arity, typename Is =
            typename util::detail::make_index_pack<Arity>::type>
        struct make_fallback_signature;

        template <std::size_t /*I*/>
        struct make_fallback_argument
        {
            typedef fallback_argument type;
        };

        template <std::size_t Arity, std::size_t ...Is>
        struct make_fallback_signature<Arity, util::detail::pack_c<std::size_t, Is...> >
        {
            typedef fallback_call const& type(
                typename make_fallback_argument<Is>::type...);
        };

        ///////////////////////////////////////////////////////////////////////
        template <typename T, std::size_t Arity>
        struct callable_wrapper_fallback
        {
            typedef typename make_fallback_signature<Arity>::type* function_ptr;

            operator function_ptr() const volatile;
        };

        template <typename R>
        struct callable_wrapper_fallback<R(*)(), 0>
        {};

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

        template <typename R, typename ...Ps, std::size_t Arity>
        struct callable_wrapper<R(*)(Ps...), Arity>
          : callable_wrapper_fallback<R(*)(Ps...), Arity>
        {
            R operator()(Ps...) const volatile;
        };

        template <typename R, typename C, typename ...Ps, std::size_t Arity>
        struct callable_wrapper<R(C::*)(Ps...), Arity>
          : callable_wrapper_fallback<R(C::*)(Ps...), Arity>
        {
            R operator()(C*, Ps...) const volatile;
            R operator()(C&, Ps...) const volatile;
            R operator()(C&&, Ps...) const volatile;

            template <typename T>
            typename boost::enable_if_c<
                boost::has_dereference<T, C&>::value
              , R
            >::type operator()(T, Ps...) const volatile;
        };

        template <typename R, typename C, typename ...Ps, std::size_t Arity>
        struct callable_wrapper<R(C::*)(Ps...) const, Arity>
          : callable_wrapper_fallback<R(C::*)(Ps...) const, Arity>
        {
            R operator()(C const*, Ps...) const volatile;
            R operator()(C const&, Ps...) const volatile;
            R operator()(C const&&, Ps...) const volatile;
            template <typename T>
            typename boost::enable_if_c<
                boost::has_dereference<T, C const&>::value
              , R
            >::type operator()(T, Ps...) const volatile;
        };

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
        template <typename F, typename ...Ts>
        struct is_callable_impl
        {
            typedef typename util::detail::qualify_as<
                    detail::callable_wrapper<
                        typename util::decay<F>::type, sizeof...(Ts)
                    >, F
                >::type callable_wrapper;
            typedef char (&no_type)[1];
            typedef char (&yes_type)[2];

            template<typename T>
            static yes_type can_be_called(T const &);
            static no_type can_be_called(detail::fallback_call const &);

            static bool const value =
                sizeof(can_be_called((boost::declval<callable_wrapper>()(
                    boost::declval<Ts>()...), 0))
                ) == sizeof(yes_type);
            typedef boost::mpl::bool_<value> type;
        };
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename T>
    struct is_callable;

    template <typename F, typename ...Ts>
    struct is_callable<F(Ts...)>
      : detail::is_callable_impl<F, Ts...>
    {};
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
                typename util::decay_unwrap<F>::type&(A...)
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

namespace hpx { namespace traits { namespace detail
{
    ///////////////////////////////////////////////////////////////////////////
    template <typename T>
    struct is_deferred_callable;

    template <typename F, typename ...Ts>
    struct is_deferred_callable<F(Ts...)>
      : is_callable<
            typename util::decay<F>::type(typename util::decay<Ts>::type...)
        >
    {};
}}}

#endif
