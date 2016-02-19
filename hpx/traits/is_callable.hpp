//  Copyright (c) 2007-2012 Hartmut Kaiser
//  Copyright (c) 2013-2015 Agustin Berge
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_TRAITS_IS_CALLABLE_HPP
#define HPX_TRAITS_IS_CALLABLE_HPP

#include <hpx/config.hpp>
#include <hpx/util/always_void.hpp>
#include <hpx/util/decay.hpp>
#include <hpx/util/result_of.hpp>

#include <boost/type_traits/integral_constant.hpp>

#include <type_traits>

namespace hpx { namespace traits
{
    ///////////////////////////////////////////////////////////////////////////
    namespace detail
    {
        template <typename T, typename R, typename Enable = void>
        struct is_callable_impl
          : boost::false_type
        {};

        template <typename T>
        struct is_callable_impl<T, void,
            typename util::always_void<typename util::result_of<T>::type>::type
        > : boost::true_type
        {};

        template <typename T, typename R>
        struct is_callable_impl<T, R,
            typename util::always_void<typename util::result_of<T>::type>::type
        > : boost::integral_constant<bool,
                std::is_convertible<
                    typename util::result_of<T>::type,
                    R
                >::value
            >
        {};
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename T, typename R = void>
    struct is_callable;

    template <typename F, typename ...Ts, typename R>
    struct is_callable<F(Ts...), R>
      : detail::is_callable_impl<F(Ts...), R>
    {};

    ///////////////////////////////////////////////////////////////////////////
    namespace detail
    {
        template <typename T>
        struct is_deferred_callable;

        template <typename F, typename ...Ts>
        struct is_deferred_callable<F(Ts...)>
          : is_callable<
                typename util::decay_unwrap<F>::type(
                    typename util::decay_unwrap<Ts>::type...)
            >
        {};
    }
}}

#endif
