//  Copyright (c) 2007-2012 Hartmut Kaiser
//  Copyright (c) 2013-2015 Agustin Berge
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// hpxinspect:nodeprecatedname:is_callable
// hpxinspect:nodeprecatedname:util::result_of

#ifndef HPX_TRAITS_IS_CALLABLE_HPP
#define HPX_TRAITS_IS_CALLABLE_HPP

#include <hpx/config.hpp>
#include <hpx/type_support/always_void.hpp>
#include <hpx/util/result_of.hpp>

#include <type_traits>

namespace hpx { namespace traits
{
    ///////////////////////////////////////////////////////////////////////////
    namespace detail
    {
        template <typename T, typename R, typename Enable = void>
        struct is_callable_impl
          : std::false_type
        {};

        template <typename T>
        struct is_callable_impl<T, void,
            typename util::always_void<typename util::result_of<T>::type>::type
        > : std::true_type
        {};

        template <typename T, typename R>
        struct is_callable_impl<T, R,
            typename util::always_void<typename util::result_of<T>::type>::type
        > : std::is_convertible<
                typename util::result_of<T>::type,
                R
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
    template <typename F, typename ...Ts>
    struct is_invocable
      : detail::is_callable_impl<F&&(Ts&&...), void>
    {};

    template <typename R, typename F, typename ...Ts>
    struct is_invocable_r
      : detail::is_callable_impl<F&&(Ts&&...), R>
    {};
}}

#endif
