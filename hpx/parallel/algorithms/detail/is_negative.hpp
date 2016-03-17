//  Copyright (c) 2007-2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PARALLEL_DETAIL_IS_NEGATIVE_JUL_2014_01_0148PM)
#define HPX_PARALLEL_DETAIL_IS_NEGATIVE_JUL_2014_01_0148PM

#include <boost/utility/enable_if.hpp>
#include <boost/type_traits/is_signed.hpp>
#include <boost/type_traits/is_unsigned.hpp>

namespace hpx { namespace parallel { HPX_INLINE_NAMESPACE(v1) { namespace detail
{
    // main template represents non-integral types (raises error)
    template <typename Size, typename Enable = void>
    struct is_negative_helper;

    // signed integral values may be negative
    template <typename T>
    struct is_negative_helper<T,
        typename boost::enable_if<boost::is_signed<T> >::type>
    {
        static bool call(T const& size) { return size < 0; }

        static T abs(T const& val) { return val < 0 ? -val : val; }

        static T negate(T const& val)
        {
            return -val;
        }
    };

    // unsigned integral values are never negative
    template <typename T>
    struct is_negative_helper<T,
        typename boost::enable_if<boost::is_unsigned<T> >::type>
    {
        static bool call(T const&) { return false; }

        static T abs(T const& val) { return val; }

        static T negate(T const& val)
        {
            return val;
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename T>
    bool is_negative(T const& val)
    {
        return is_negative_helper<T>::call(val);
    }

    template <typename T>
    T abs(T const& val)
    {
        return is_negative_helper<T>::abs(val);
    }

    template <typename T>
    T negate(T const& val)
    {
        return is_negative_helper<T>::negate(val);
    }
}}}}

#endif


