//  Copyright (c) 2007-2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PARALLEL_DETAIL_IS_NEGATIVE_JUL_2014_01_0148PM)
#define HPX_PARALLEL_DETAIL_IS_NEGATIVE_JUL_2014_01_0148PM

#include <hpx/config.hpp>
#include <hpx/parallel/config/inline_namespace.hpp>

#include <type_traits>

namespace hpx { namespace parallel { HPX_INLINE_NAMESPACE(v1) { namespace detail
{
    // main template represents non-integral types (raises error)
    template <typename Size, typename Enable = void>
    struct is_negative_helper;

    // signed integral values may be negative
    template <typename T>
    struct is_negative_helper<T,
        typename std::enable_if<std::is_signed<T>::value>::type>
    {
        HPX_HOST_DEVICE HPX_FORCEINLINE
        static bool call(T const& size) { return size < 0; }

        HPX_HOST_DEVICE HPX_FORCEINLINE
        static T abs(T const& val) { return val < 0 ? -val : val; }

        HPX_HOST_DEVICE HPX_FORCEINLINE
        static T negate(T const& val)
        {
            return -val;
        }
    };

    // unsigned integral values are never negative
    template <typename T>
    struct is_negative_helper<T,
        typename std::enable_if<std::is_unsigned<T>::value>::type>
    {
        HPX_HOST_DEVICE HPX_FORCEINLINE
        static bool call(T const&) { return false; }

        HPX_HOST_DEVICE HPX_FORCEINLINE
        static T abs(T const& val) { return val; }

        HPX_HOST_DEVICE HPX_FORCEINLINE
        static T negate(T const& val)
        {
            return val;
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename T>
    HPX_HOST_DEVICE HPX_FORCEINLINE bool is_negative(T const& val)
    {
        return is_negative_helper<T>::call(val);
    }

    template <typename T>
    HPX_HOST_DEVICE HPX_FORCEINLINE T abs(T const& val)
    {
        return is_negative_helper<T>::abs(val);
    }

    template <typename T>
    HPX_HOST_DEVICE HPX_FORCEINLINE T negate(T const& val)
    {
        return is_negative_helper<T>::negate(val);
    }
}}}}

#endif


