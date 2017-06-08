//  Copyright (c) 2013-2015 Agustin Berge
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// hpxinspect:nodeprecatedinclude:boost/ref.hpp
// hpxinspect:nodeprecatedname:boost::reference_wrapper

#ifndef HPX_UTIL_RESULT_OF_HPP
#define HPX_UTIL_RESULT_OF_HPP

#include <hpx/config.hpp>

#include <boost/ref.hpp>

#include <type_traits>
#include <utility>

namespace hpx { namespace util
{
    namespace detail
    {
        ///////////////////////////////////////////////////////////////////////
        // f(t0, t1, ..., tN)
#if HPX_HAS_CXX14_LIB_RESULT_OF_SFINAE
        template <typename T>
        struct result_of_function_object
          : std::result_of<T>
        {};
#else
        namespace result_of_function_object_impl
        {
            struct fallback
            {
                template <typename T>
                fallback(T const&){}
            };

            template <typename ...Ts>
            fallback invoke(fallback, Ts&&...);

            template <typename F, typename ...Ts>
            decltype(std::declval<F>()(std::declval<Ts>()...))
            invoke(F&&, Ts&&...);

            template <typename T>
            struct result_of_invoke;

            template <typename F, typename ...Ts>
            struct result_of_invoke<F(Ts...)>
            {
                typedef decltype(result_of_function_object_impl::invoke(
                    std::declval<F>(), std::declval<Ts>()...)) type;
            };

            template <typename T, typename R = typename result_of_invoke<T>::type>
            struct result_of_function_object
            {
                typedef R type;
            };

            template <typename T>
            struct result_of_function_object<T, fallback>
            {};
        }
        using result_of_function_object_impl::result_of_function_object;
#endif

        ///////////////////////////////////////////////////////////////////////
        template <typename T>
        struct result_of_member_pointer_impl;

        // t0.*f
        template <typename R, typename C>
        struct result_of_member_pointer_impl<R C::*>
        {
            R& operator()(C&);
            R const& operator()(C const&);
            R&& operator()(C&&);
            R const&& operator()(C const&&);
        };

        // (t0.*f)(t1, ..., tN)
        template <typename R, typename C, typename ...Ps>
        struct result_of_member_pointer_impl<R (C::*)(Ps...)>
        {
            R operator()(C&, Ps...);
            R operator()(C&&, Ps...);
        };

        template <typename R, typename C, typename ...Ps>
        struct result_of_member_pointer_impl<R (C::*)(Ps...) const>
        {
            R operator()(C const&, Ps...);
            R operator()(C const&&, Ps...);
        };

        ///////////////////////////////////////////////////////////////////////
        namespace has_dereference_impl
        {
            struct fallback
            {
                template <typename T>
                fallback(T const&){}
            };

            fallback operator*(fallback);

            template <typename T>
            struct has_dereference
            {
                static bool const value =
                    !std::is_same<decltype(*std::declval<T>()), fallback>::value;
            };
        }
        using has_dereference_impl::has_dereference;

        template <typename C, typename T, typename Enable = void>
        struct result_of_member_pointer
        {};

        // t0.*f, (t0.*f)(t1, ..., tN)
        template <typename C, typename F, typename T0, typename ...Ts>
        struct result_of_member_pointer<C, F(T0, Ts...),
            typename std::enable_if<
                std::is_base_of<C, typename std::decay<T0>::type>::value
            >::type
        > : result_of_function_object<
                result_of_member_pointer_impl<typename std::decay<F>::type>(
                    T0, Ts...)
            >
        {};

        // (*t0).*f, ((*t0).*f)(t1, ..., tN)
        template <typename C, typename F, typename T0, typename ...Ts>
        struct result_of_member_pointer<C, F(T0, Ts...),
            typename std::enable_if<
                std::enable_if<
                    !std::is_base_of<C, typename std::decay<T0>::type>::value
                  , has_dereference<T0>
                >::type::value
            >::type
        > : result_of_function_object<
                result_of_member_pointer_impl<typename std::decay<F>::type>(
                    decltype(*std::declval<T0>()), Ts...)
            >
        {};

        ///////////////////////////////////////////////////////////////////////
        template <typename FD, typename T>
        struct result_of_impl
          : result_of_function_object<T>
        {};

        template <typename M, typename C, typename F, typename ...Ts>
        struct result_of_impl<M C::*, F(Ts...)>
          : result_of_member_pointer<C, M C::*(Ts...)>
        {};

        template <typename R, typename C, typename ...Ps, typename F, typename ...Ts>
        struct result_of_impl<R (C::*)(Ps...), F(Ts...)>
          : result_of_member_pointer<C, R (C::*(Ts...))(Ps...)>
        {};

        template <typename R, typename C, typename ...Ps, typename F, typename ...Ts>
        struct result_of_impl<R (C::*)(Ps...) const, F(Ts...)>
          : result_of_member_pointer<C, R (C::*(Ts...))(Ps...) const>
        {};

        // support boost::[c]ref, which is not callable as std::[c]ref
        template <typename X, typename F, typename ...Ts>
        struct result_of_impl< ::boost::reference_wrapper<X>, F(Ts...)>
          : result_of_impl<X, X&(Ts...)>
        {};
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename T>
    struct result_of;

    template <typename F, typename ...Ts>
    struct result_of<F(Ts...)>
      : detail::result_of_impl<typename std::decay<F>::type, F(Ts...)>
    {};

    ///////////////////////////////////////////////////////////////////////////
    template <typename F, typename ...Ts>
    struct invoke_result
      : detail::result_of_impl<typename std::decay<F>::type, F&&(Ts&&...)>
    {};
}}

#endif
